import sys
import os
import parasail
import numpy as np
import re
import argparse
import pydivsufsort

def options():
	parser = argparse.ArgumentParser(description='Modernized target site duplication searcher. Takes two short genome sequences \
												which you believe contain a TSD and quickly searches for shared subsequences, allowing for gaps.\
												\tThis code is primarily meant to be called from within another program rather than being directly used\
												on the commandline, but we support it anyway.')
	# Method options
	parser.add_argument(
		'--sequences',
		default=None,
		help='File of sequences in FASTA format to search for TSDs in.'
	)
	
	# Method options
	parser.add_argument(
		'--output',
		default=None,
		help='Output file to report results in. If none, will print to stdout'
	)
	
	parser.add_argument(
		'--left_window',
		type=int,
		default=70,
		help='The first [--left_window] (default: 70) bp of each sequence in your input will searched for TSDs against the last [--right_window] bp'
	)
	
	parser.add_argument(
		'--right_window',
		type=int,
		default=50,
		help='The last [--right_window] (default: 50) bp of each sequence in your input will searched for TSDs against the first [--left_window] bp'

	)
	parser.add_argument(
		'--method',
		choices=['tsd_searcher', 'sinefinder'],
		default='tsd_searcher',
		help='Method to use for scoring TSD candidates (default: tsd_searcher)'
	)

	# TSD searcher and sinefinder options
	parser.add_argument(
		'--min-ok-length',
		type=int,
		default=10,
		help='Minimum length of TSD (default: 10)'
	)

	parser.add_argument(
		'--max-mismatch',
		type=int,
		default=2,
		help='Maximum number of total mismatches allowed under tsd_searcher method (default: 2)'
	)

	parser.add_argument(
		'--poly_at-ok',
		action='store_true',
		default=False,
		help='Allow polyA/T sequences to be returned as TSDs (default: False)'
	)

	parser.add_argument(
		'--polyat-threshold',
		type=float,
		default=1,
		help='Number of non-A/T characters allowed before a polyA/T is excluded; pointless when used with --poly_at-ok (default: 1)'
	)

	parser.add_argument(
		'--check-inverts',
		action='store_true',
		default=False,
		help='Check right sequences for TIRs (default: False)'
	)

	# Scoring options
	parser.add_argument(
		'--gap-penalty',
		type=int,
		default=5,
		help='Alignment gap penalty (default: 10)'
	)

	parser.add_argument(
		'--extension-penalty',
		type=int,
		default=1,
		help='Alignment extension penalty (default: 5)'
	)

	parser.add_argument(
		'--sf-score-thresh',
		type=int,
		default=10,
		help='TSD score threshold for sinefinder method (default: 10) TSD score = TSD length - (--sf-mismatch-penalty * num_mismatches)'
	)

	parser.add_argument(
		'--sf-mismatch-thresh',
		type=int,
		default=2,
		help='Maximum number of mismatches and gaps allowed in a TSD for sinefinder method (default: 2)'
	)

	parser.add_argument(
		'--sf-mismatch-penalty',
		type=int,
		default=1,
		help='Score penalty for each mismatch for sinefinder method (default: 1).'
	)

	parser.add_argument(
		'--return-best-only',
		action='store_true',
		default=True,
		help='Return only the single best TSD candidate for a sequence, if any (default: True). The best match is defined as whichever TSD \
		minimizes distance between the left and right TSD loci. Ties are broken by an unweighted TSD score of TSD length-num_mismatches. \
		Ties for ties favors first candidate. If --check-inverts is used, only a forward or a reverse sequence will be returned, not one of each.'
	)
	
	args = parser.parse_args()
	return parser, args

class alignment_tsd_tir_finder:
	def __init__(self, method = 'tsd_searcher', min_ok_length = 10, max_mismatch = 1, polyAT_ok = False, polyAT_threshold = 1, 
				check_inverts = False, gap_penalty = 1, extension_penalty = 0, sf_score_thresh = 10, 
				sf_mismatch_thresh = 2, sf_mismatch_penalty = 1, lookaround = 10, return_best_only = True):
				
		self.method = method
		self.revcmp_table = str.maketrans('ACGTacgt', 'TGCAtgca')
		
		self.np_encoding = {'-':0, 'A':1, 'C':2, 'G':3, 'T':4}
		self.np_decoding = {0:'-', 1:'A', 2:'C', 3:'G', 4:'T'}
		
		self.min_ok_length = min_ok_length
		self.max_mismatch = max_mismatch
		self.max_consecutive_mismatches = 1
		
		self.lookaround_neighborhood = lookaround
		
		self.sf_mm = sf_mismatch_thresh
		self.sf_pen = sf_mismatch_penalty
		self.sf_score = sf_score_thresh
		
		self.polyAT_ok = polyAT_ok
		self.polyAT_threshold = polyAT_threshold
		
		self.check_inverts = check_inverts
		self.gap_penalty = gap_penalty
		self.ext_penalty = extension_penalty
		
		self.best = return_best_only
		
		self.candidates = []

	#To find inverted repeats, revcmp the right string and realign
	def revcomp(self, sequence):
		sequence = sequence[::-1]
		sequence = sequence.translate(self.revcmp_table)
		return sequence

	#Convert a string to a numeric representation to make vector ops a little easier to work with in python
	def encode_numpy(self, sequence):
		sequence = re.sub('[^ATCGatcg]', '-', sequence)
		sequence = np.array([self.np_encoding[c] for c in sequence], dtype = np.int32)
		return sequence

	def decode_numpy(self, sequence):
		converted = ''.join([self.np_decoding[c] for c in sequence])
		return converted

	#Vectorized numpy run-length encoding function;
	#Credit to #https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
	#Here, used to encode runs of matching and mismatching characters from encoded numeric numpy array
	def rle(self, ia):
		""" run length encoding. Partial credit to R rle function. 
			Multi datatype arrays catered for including non Numpy
			returns: tuple (runlengths, startpositions, values) """
		n = len(ia)
		if n == 0: 
			return None, None, None
		else:
			y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
			i = np.append(np.where(y), n - 1)   # must include last element posi
			z = np.diff(np.append(-1, i))       # run lengths
			p = np.cumsum(np.append(0, z))[:-1] # positions
			
			arr = np.vstack([ia[i], p, z]).T
			
			return arr

	#Suffix array and longest common prefix array based approach to find seed elements of min length minimum_length
	def find_longest_shared_subsequences(self, l, r, minimum_length = 5):
		#Join seqs separated by '#' char
		js = f'{l}#{r}$'

		#Create suffix array
		sa = pydivsufsort.divsufsort(js)
		
		#Crearte longest common prefix array
		lcp = pydivsufsort.kasai(js, sa)
		
		shared_substrings = []
		n1 = len(l)
		
		#Iterate over common prefixes to find repeats
		for i in range(0, len(lcp)-1):
			#If the two suffixes originate from l and r, not both from l or both from r
			#The substrings are a repeat if they appear twice consecutively in sa.
			if (sa[i + 1] < n1) != (sa[i] < n1):
				#If the shared substring is long enough
				if lcp[i] >= minimum_length:
					shared_substring_size = lcp[i]
					
					#These are not guaranteed to correspont to l or r; the lower corresponds to l, and the higher - 1 - n1 corresponds to r
					start_indices = [sa[i], sa[i + 1]]
					next_row = (min(start_indices), max(start_indices)-(n1+1), shared_substring_size)
					
					shared_substrings.append(next_row)
		
		if len(shared_substrings) > 0:
			
			shared_substrings = np.array(shared_substrings)
			
			#Unfortunately, because this is all prefixes and they are not sorted by a length or position logic, we have cleaning to do
			
			#Sort by start location, then pattern length
			shared_substrings = shared_substrings[np.lexsort((-shared_substrings[:,1], shared_substrings[:, 0],))]
			
			removal_sub = np.array([-1, -1, 1])
			old_subs = -1
			
			#Sometimes odd ordering prevents this from working on just one pass.
			while old_subs != shared_substrings.shape[0]:
			
				old_subs = shared_substrings.shape[0]
				
				current_row = None
				#If the next row is a substring of the previous row, the start indices will be +1 relative to the previous row and the pattern length will be -1
				pattern_gaps = shared_substrings[:-1] - shared_substrings[1:]
				
				#First record is always kept
				filterer = np.ones(shape = shared_substrings.shape[0], dtype = np.bool_)
				#For other records, check if they are a repeat pattern based on similarity to removal_sub
				filterer[1:] = np.all(pattern_gaps != removal_sub, axis = 1)
				
				#Remove substrings that are substrings of another string in the list
				shared_substrings = shared_substrings[filterer]

		else:
			#First search fail condition
			shared_substrings = None
			
		return shared_substrings
	
	#Given an exact match, look ahead and behind in both query and ref by no more than max_lookaround and align using exact match as anchor;
	#Align the left and right sequences; use semi-global alignment with no start penalty left and no end penalty right
	#So that the sequences are liable to align at the near ends
	
	'''
	How we do:
	(might be able to use parasail cigar instead of direct sequences? A little complicated left)
	
	(1) For each exact match
	
		(2) Extend right by max_lookaround
		(3) Align the right sequences using the exact match as an anchor; semi global no end penalty (matches closer to exact match preferred)
		(4) Iterate right until max_mismatches or score condition fail or end of either seq
		(4b) Back to 2, another extension of max_lookaround if matches continue
		
		(5) Extend left by max_lookaround
		(6) Align the left sequences using the exact match as an acnchor; semi global no start penalty (matches closer to exact match preferred)
		(7) Iterate left until max_mismatches or score condition fail or end of either seq
		(7b) Back to 5, another extension of max_lookaround is matches continue
		
		(8) Join left and right matching sequence indices
		(9) Extract the highest scoring run of sequences which:
			(a) Starts and ends with matches
			(b) Exceeds score conditions
		
	'''
	
	def find_similar_sequences_lookaround(self, left, right, exact_matches):
		llen, rlen = len(left), len(right)
		
		extensions_numeric = []
		extensions_text = []
		
		for i in exact_matches:
			left_start = i[0]
			right_start = i[1]
			pattern_length = i[2]
			left_end = left_start + pattern_length
			right_end = right_start + pattern_length
			
			exact_repeat = left[left_start:left_end]
			
			right_extend_left_aln = ''
			right_extend_right_aln = ''
			left_extend_left_aln = ''
			left_extend_right_aln = ''
			
			continue_search = True
			
			####Right side search#####
			
			ml = self.lookaround_neighborhood
			loop_num = 0
			num_mismatch = 0
			num_match = 0
			last_match = 0
			while continue_search:
				#No infinite loops, please
				if self.lookaround_neighborhood < 1:
					continue_search = False
					break
				loop_num += 1
				#Check boundary
				if left_end + ml > llen:
					ml = llen - left_end
					continue_search = False
				if right_end + ml > rlen:
					ml = rlen - right_end
					continue_search = False
					
				#Collect exact repeat and sequence to its right
				sub_left   = left[left_start:left_end+ml]
				sub_right = right[right_start:right_end+ml]
				
				res = parasail.sg_qe_de_trace_striped_sat(sub_left, sub_right, self.gap_penalty, self.ext_penalty, parasail.blosum62)
				
				left_aln = res.traceback.query
				right_aln = res.traceback.ref

				intact_anchor_left = left_aln.startswith(exact_repeat)
				intact_anchor_right = right_aln.startswith(exact_repeat)

				#The alignment doesn't even begin with the exact repeat; it is of low quality and not worth checking further
				if not (intact_anchor_left and intact_anchor_right):
					continue_search = False
				else:
					
					extend_right = 0
					last_match = 0
					num_mismatch = 0
					num_match = 0
					
					#Extend right along alignments until max mismatch reached
					for qc, rc in zip(list(left_aln[pattern_length:]), list(right_aln[pattern_length:])):
						extend_right += 1
						if qc != rc:
							num_mismatch += 1
							#Can't continue adding sequence
							if num_mismatch >= self.max_mismatch:
								continue_search = False
								break
						else:
							num_match += 1
							last_match = extend_right
							
					#Don't bother grabbing sequence if there were no matches by last mismatch
					if last_match > 0:
						right_extend_left_aln = left_aln[pattern_length:pattern_length+last_match]
						right_extend_right_aln = right_aln[pattern_length:pattern_length+last_match]
					
				ml += self.lookaround_neighborhood
			
			right_mm = num_mismatch
			right_mat = num_match
			
			right_extension_size = last_match
			
			#Reset for left search
			continue_search = True
			
			####Left side search#####
			
			ml = self.lookaround_neighborhood
			loop_num = 0
			num_mismatch = 0
			num_match = 0
			last_match = 0
			while continue_search:
				#No infinite loops, please
				if self.lookaround_neighborhood < 1:
					continue_search = False
					break
				loop_num += 1
				#Check boundary
				if ml > left_start:
					ml = left_start
					continue_search = False
				if ml > right_start:
					ml = right_start
					continue_search = False
					
				#Collect sequence to the left of the exact repeat and the exact repeat
				sub_left   = left[left_start- ml:left_end]
				sub_right = right[right_start- ml:right_end]
				
				res = parasail.sg_qb_db_trace_striped_sat(sub_left, sub_right, self.gap_penalty, self.ext_penalty, parasail.blosum62)
				
				left_aln = res.traceback.query
				right_aln = res.traceback.ref
				
				intact_anchor_left = left_aln.endswith(exact_repeat)
				intact_anchor_right = right_aln.endswith(exact_repeat)
				
				#The alignment doesn't even begin with the exact repeat; it is of low quality and not worth checking further
				if not (intact_anchor_left and intact_anchor_right):
					continue_search = False
				else:
				
					#Flip the direction of the alignments to that progression is functionally right -> left
					left_aln = left_aln[::-1]
					right_aln = right_aln[::-1]
					
					extend_left = 0
					last_match = 0
					num_mismatch = 0
					num_match = 0
					
					#Extend left along alignments until max mismatch reached, 
					for qc, rc in zip(list(left_aln[pattern_length:]), list(right_aln[pattern_length:])):
						extend_left += 1
						if qc != rc:
							num_mismatch += 1
							#Can't continue adding sequence
							if num_mismatch >= self.max_mismatch:
								continue_search = False
								break
						else:
							last_match = extend_left
					
					
					#Don't bother grabbing sequence if there were no matches by last mismatch
					if last_match > 0:
						left_extend_left_aln = left_aln[pattern_length:pattern_length+last_match]
						left_extend_right_aln = right_aln[pattern_length:pattern_length+last_match]
						#Correct the direction again
						left_extend_left_aln = left_extend_left_aln[::-1]
						left_extend_right_aln = left_extend_right_aln[::-1]
					
					
				ml += self.lookaround_neighborhood
				
			left_mm = num_mismatch
			left_mat = num_match
			
			left_extension_size = last_match
			
			#actual position offsets after removing gap char
			ll_gaps = left_extend_left_aln.count('-')
			lr_gaps = left_extend_right_aln.count('-')
			rl_gaps = right_extend_left_aln.count('-')
			rr_gaps = right_extend_right_aln.count('-')
			
			bases_left_left = len(left_extend_left_aln)-ll_gaps
			bases_left_right = len(left_extend_right_aln)-lr_gaps
			bases_right_left = len(right_extend_left_aln)-rl_gaps
			bases_right_right = len(right_extend_right_aln)-rr_gaps
			
			my_extension_numeric = (left_extension_size, left_mat, left_mm, ll_gaps, lr_gaps, bases_left_left, bases_left_right, right_extension_size, right_mat, right_mm, rl_gaps, rr_gaps, bases_right_left, bases_right_right,)
			my_extension = (left_extend_left_aln, left_extend_right_aln, exact_repeat, right_extend_left_aln, right_extend_right_aln)
			
			extensions_text.append(my_extension)
			extensions_numeric.append(my_extension_numeric)
			
		
		extensions_numeric = np.array(extensions_numeric)
			
		return extensions_numeric, extensions_text


	def find_valid_segments(self, runs):
		segments = []
		n = runs.shape[0]
		for i in range(n):
			if runs[i][0] != 1:
				continue
			false_sum = 0
			if false_sum == self.max_mismatch:
				segments.append(runs[i:i+1])
			for j in range(i + 1, n):
				if runs[j][0] == 0:
					false_sum += runs[j][2]
					if false_sum > self.max_mismatch:
						break
				else:
					if false_sum == self.max_mismatch or j == n - 1:
						segments.append(runs[i:j+1])
		
		return segments

	#Find the highest scoring substring for each extension candidate
	def score_extensions(self, extension_strings, prefer_exact = True):
		winning_candidates = []
		for candidate in extension_strings:
		
			exact_repeat = candidate[2]
			exact_score = len(exact_repeat)
			exact_mismatch = 0
			
			query  = f'{candidate[0]}{exact_repeat}{candidate[3]}'
			target = f'{candidate[1]}{exact_repeat}{candidate[4]}'
			
			
			
			#This is an excellent point to check for polyAT
			
			#No extension found
			if query == exact_repeat:
				winning_candidates.append((exact_repeat, exact_repeat, exact_score, exact_mismatch,))
			else:
				query = self.encode_numpy(query)
				target = self.encode_numpy(target)
				all_eq = query == target
				
				#Array of [[0/1 (false/true), start_index_in_all_eq, run_length_of_true_or_false]]
				rle_array = self.rle(all_eq)
				
				#This is indicative of some strange behavior where a non-full length exact repeat was recovered 
				#by the suffix array code. It's OK, we can handle it here.
				if rle_array.shape[0] == 1:
					#print('Strange behavior')
					if rle_array[0][0] == 1:
						score = rle_array[0][2]
						mismatch = 0
						winning_candidates.append((self.decode_numpy(query), self.decode_numpy(target), int(score), int(mismatch),))
				
				else:
					#This is true, false, true; only one valid substring is acceptable
					if rle_array.shape[0] == 3:
						mismatch = np.sum(rle_array[:,2][rle_array[:,0] == 0])
						score = len(query) - mismatch ** 2
						
						if score > exact_score:
							winning_candidates.append((self.decode_numpy(query), self.decode_numpy(target), int(score), int(mismatch),))
						else:
							winning_candidates.append((exact_repeat, exact_repeat, exact_score, exact_mismatch,))
						
					else:
						#Find subarrays that start and end with a true and internally contain no more than self.max_mismatch mismatches
						candidate_runs = self.find_valid_segments(rle_array)
						
						best_score = 0
						best_mismatch = 0
						winning_candidate = candidate_runs[0]
						
						for candidate in candidate_runs:
							mismatch = np.sum(candidate[:,2][candidate[:,0] == 0])
							#Total trues = trues + falses - falses; score = total_trues - mismatch^2
							score = np.sum(candidate[:,2]) - mismatch - (mismatch**2)
							if score > best_score:
								best_score = score
								winning_candidate = candidate
								best_mismatch = mismatch
						
						#Check if extension was successful
						if best_score > exact_score:
							winning_candidates.append((self.decode_numpy(query), self.decode_numpy(target), int(best_score), int(best_mismatch),))
						else:
							winning_candidates.append((exact_repeat, exact_repeat, exact_score, exact_mismatch,))
	
		return winning_candidates


	#If requested, return only the best matching TSD
	#Biologically, this must be the closest to the original candidate, irrespective of length
	def get_best_hit(self, left_seqlen):
		if len(self.candidates) > 0:
			winning_index = 0
			winning_tsd_length = 0
			winning_tsd_mismatch = 0
			idx = 0
			winning_distance = 1_000_000_000 #An unlikely distance for a TSD, to say the least
			for candidate in self.candidates:
				lseq, lst, lend, rseq, rst, rend, tsd_length, tsd_mismatches, orient = candidate
				left_distance_to_end = left_seqlen - lend
				distance = left_distance_to_end + rst #right start is always relative to 0
				
				#Strictly closer just wins
				if distance < winning_distance:
					winning_distance = distance
					winning_index = idx
					winning_tsd_length = tsd_length
					winning_tsd_mismatch = tsd_mismatches
				
				else:
					#Tiebreak favor less gappy and or longer TSD
					if distance == winning_distance:
						if tsd_length - tsd_mismatches > winning_tsd_length - winning_tsd_mismatch:
							winning_distance = distance
							winning_index = idx
							winning_tsd_length = tsd_length
							winning_tsd_mismatch = tsd_mismatches
					
				idx += 1
				
			self.candidates = [self.candidates[winning_index]]
	
	#Sequence alignment based approach using parasail
	def tsd_by_sequence_alignment(self, left_seq, right_seq):
		self.candidates = []
		forward = None
		reverse = None
		#Low penalty semi-global sequence alignment to find repeats within substrings
		res = parasail.sg_trace_striped_sat(left_seq, right_seq, self.gap_penalty, self.ext_penalty, parasail.blosum62)
		
		#Fortunately this never encodes a double '-', so we can ignore a match on that character - it will always be false
		left = res.traceback.query
		right = res.traceback.ref
		
		if self.method == 'tsd_searcher':
			self.find_similar_sequences_tsd_searcher(left, right, is_forward = True)
		if self.method == 'sinefinder':
			self.find_similar_sequences_sinefinder(left, right, is_forward = True)

		#Look for TIRs
		if self.check_inverts:
			#Save forward sequences, if any
			forward_sequences = self.candidates
			self.candidates = []
			rsl = len(right_seq)
			og_rseq = right_seq
			right_seq = self.revcomp(right_seq)
			#Low penalty semi-global sequence alignment to find repeats within substrings
			#Semi-global doesn't penalize gaps and mismatches at start and end of aln, and we don't care about those in this context.
			res = parasail.sg_trace_striped_sat(left_seq, right_seq, self.gap_penalty, self.ext_penalty, parasail.blosum62)
			#Low penalty local sequence alignment to find repeates within substrings
			#res = parasail.sw_trace_striped_sat(left_seq, right_seq, self.gap_penalty, self.ext_penalty, parasail.blosum62)
			
			#Fortunately this never encodes a double '-', so we can ignore a match on that character - it will always be false
			left = res.traceback.query
			right = res.traceback.ref
						
			if self.method == 'tsd_searcher':
				self.find_similar_sequences_tsd_searcher(left, right, is_forward = False)
			if self.method == 'sinefinder':
				self.find_similar_sequences_sinefinder(left, right, is_forward = False)
			
			reverse_candidates_clean = []
			#If any reverse candidate, flip their orientation back to forward strand logic
			if len(self.candidates) > 0:
				for candidate in self.candidates:
					lseq, lst, lend, rseq, rst, rend, tsd_length, tsd_mismatches, orient = candidate
					#Flip the right sequence, start and stop indices to forward orientation
					if rseq is not None:
						fst, fend = rsl-rend, rsl-rst
						rseq = self.revcomp(rseq)
							
						f_orient_candidate = (lseq, lst, lend, rseq, fst, fend, tsd_length, tsd_mismatches, orient,)
						reverse_candidates_clean.append(f_orient_candidate)
						
			#Reset candidates list with forward
			self.candidates = forward_sequences
			#Add corrected inverses, if any
			self.candidates.extend(reverse_candidates_clean)
			
		
		if self.best:
			self.get_best_hit(len(left_seq))
		
def main():
	import pyfastx
	parser, opts = options()
	
	if opts.sequences is None:
		print('You must supply a set of sequences to search for TSDs using --sequences')
	else:
		fa = pyfastx.Fasta(opts.sequences)
		
		output = opts.output
		if output is not None:
			out = open(output, 'w')
			print('sequence_id', 'left_tsd', 'left_tsd_start_ungapped', 'left_tsd_end_ungapped', 'right_tsd', 'right_tsd_start_ungapped', 
				'right_tsd_end_ungapped', 'tsd_length', 'num_mismatches', 'forward_orientation', sep = '\t', file = out)
		else:
			out = None
			print('sequence_id', 'left_tsd', 'left_tsd_start_ungapped', 'left_tsd_end_ungapped', 'right_tsd', 'right_tsd_start_ungapped', 
				'right_tsd_end_ungapped', 'tsd_length', 'num_mismatches', 'forward_orientation', sep = '\t', file = out)
		
		mn = alignment_tsd_tir_finder(method = opts.method, 
									min_ok_length = opts.min_ok_length, 
									max_mismatch = opts.max_mismatch, 
									polyAT_ok = opts.poly_at_ok, 
									polyAT_threshold = opts.polyat_threshold, 
									check_inverts = opts.check_inverts, 
									gap_penalty = opts.gap_penalty, 
									extension_penalty = opts.extension_penalty, 
									sf_score_thresh = opts.sf_score_thresh, 
									sf_mismatch_thresh = opts.sf_mismatch_thresh, 
									sf_mismatch_penalty = opts.sf_mismatch_penalty, 
									lookaround = 10,
									return_best_only = opts.return_best_only)
		
		terminate = 0
		for seq in fa:
			sequence = seq.seq.upper()
			l = sequence[0:opts.left_window]
			r = sequence[-opts.right_window:]
			
			print(seq.description)
			
			#Find the start loci in l and r of all longest exactly shared substrings of l and r and the length of that shared substring
			#A specific substring may repeat twice in this list, but it must be in a different location to do so
			#Returns array of [start_left, start_right, shared_pattern_length] or None if no shared sequences of adequate length
			shared_substrings = mn.find_longest_shared_subsequences(l, r, minimum_length = 5)

			if shared_substrings is not None:
				#Using the max-length shared substrings as seeds, use sequence alignment to attempt extending the sequences from their ends 
				#up to a certain number of mismatches before quitting
			
				'''
				Extension loci is an array of the same number of rows as shared substrings
				
				Rows are divided into two similar chunks:
				
				(1) left_extension_size, left_mat, left_mm, ll_gaps, lr_gaps, bases_left_left, bases_left_right
				(2) right_extension_size, right_mat, right_mm, rl_gaps, rr_gaps, bases_right_left, bases_right_right,
				
				extension_size is the number of characters of the alignment including gaps and mismatches
				left_mat/right_mat are the number of match characters
				left/right_mm are number of mismatch characters
				ll/lr_gaps are the number of gap characters in the alignment of the left-extended sequence for query, ref respectively
				rl/rr_gaps are the same for the right-extended sequence
				bases_left_left/left_right are the number of ungapped bases to collect on the left of the exact match
				bases_right_left/right_right are the number of ungapped bases to collect on the right of the exact match
				
				Extension strings is a list of tuples
				
				Each tuple is (left_align_string_query, left_align_string_ref, shared_repeat, right_align_string_query, right_align_string_ref)
				
				The fully extended match can be constructed as:
				left_align_string_query + shared_repeat + right_align_string_query
				left_align_string_ref   + shared_repeat + right_align_string_ref
				'''
			
				extension_loci, extension_strings = mn.find_similar_sequences_lookaround(l, r, shared_substrings)
				
				winners = mn.score_extensions(extension_strings)
				
				for wumbo in winners:
					print(wumbo)
				
				
			terminate += 1
			#if terminate == 10:
			#	break
			
			#Should I try to find adjacent runs of sequence?
			
			#Since exact matches have been found, try to align surrounding sequences
						
			
			'''
			mn.tsd_by_sequence_alignment(l, r)
			for candidate in mn.candidates:
				if out is None:
					print(seq.description, *candidate, sep = '\t')
				else:
					print(seq.description, *candidate, sep = '\t', file = out)
			'''
if __name__ == '__main__':
    main()
	
	