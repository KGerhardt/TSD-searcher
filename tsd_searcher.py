import sys
import os
import parasail
import numpy as np
import re
import argparse
import pydivsufsort
from collections import Counter

#This happens in the longest repeat checker sometimes but it's not an issue
np.seterr(divide='ignore', invalid='ignore')

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
		'--left-window',
		type=int,
		default=70,
		help='The first [--left_window] (default: 70) bp of each sequence in your input will searched for TSDs against the last [--right_window] bp'
	)
	
	parser.add_argument(
		'--right-window',
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

#Ugh, gonna need to rework the args a LOT
class alignment_tsd_tir_finder:
	def __init__(self, method = 'tsd_searcher', min_ok_length = 10, max_mismatch = 1, polyAT_TSD_ok = False, AT_rich_threshold = 1,
				check_inverts = False, gap_penalty = 1, extension_penalty = 0, sf_score_thresh = 10, 
				sf_mismatch_thresh = 2, sf_mismatch_penalty = 1, lookaround = 10, prevent_polyAT_extend = False, 
				polyAT_min_length = 5, return_best_only = True, exact_match_minsize = 5, score_alg = 'kenji'):
				
		self.method = method
		self.revcmp_table = str.maketrans('ACGTacgt', 'TGCAtgca')
		
		self.np_encoding = {'-':0, 'A':1, 'C':2, 'G':3, 'T':4}
		self.np_decoding = {0:'-', 1:'A', 2:'C', 3:'G', 4:'T'}
		
		self.exact_match_minsize = exact_match_minsize
		
		self.min_ok_length = min_ok_length
		self.max_mismatch = max_mismatch
		self.max_consecutive_mismatches = 1
		
		self.lookaround_neighborhood = lookaround
		self.prevent_polyAT_extend = prevent_polyAT_extend
		self.min_polyAT_len = polyAT_min_length
		self.polyA_definition = ''.join(['A']*self.min_polyAT_len)
		self.polyT_definition = ''.join(['T']*self.min_polyAT_len)
		
		
		self.find_polyAT = re.compile(rf'A{{{self.min_polyAT_len},}}|T{{{self.min_polyAT_len},}}')
		
		self.polyAT_TSD_ok = polyAT_TSD_ok
		self.AT_rich_threshold = AT_rich_threshold
		
		
		self.sf_mm = sf_mismatch_thresh
		self.sf_pen = sf_mismatch_penalty
		self.sf_score = sf_score_thresh
		
		
		self.check_inverts = check_inverts
		self.gap_penalty = gap_penalty
		self.ext_penalty = extension_penalty
		
		self.best = return_best_only
		
		self.candidates = []
		
		self.score_function = score_alg
		self.set_score_function()
		
	def set_score_function(self):
		#Very high mismatch penalty, heavily favor exact repeat
		def kenji_score(length, mismatches, gaps):
			if mismatches == 0 and gaps == 0:
				score = length + 2
			else:
				score = length - (mismatches + gaps) ** 2
			return score
		
		#Linear tension between TSD length and misalignment
		def sinefinder_score(length, mismatches, gaps):
			score = length - (mismatches + gaps)
		
			return score
		
		
		if self.score_function == 'kenji':
			self.score_function = kenji_score
		if self.score_function == 'sinefinder':
			self.score_function = sinefinder_score
		

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
			
			#Array of value, position, length
			arr = np.vstack([ia[i], p, z]).T
			
			return arr
	
	def is_polyAT(self, sequence, clean_sequence = False):
		seq = sequence.upper()
		masked_sequence = sequence
		seq = seq.replace('-', '')
		
		has_polyA = self.polyA_definition in seq
		has_polyT = self.polyT_definition in seq
		
		is_polyAT = has_polyA or has_polyT
		
		#Extra effort we could normally skip
		if clean_sequence:
			if is_polyAT:
				matches = re.finditer(self.find_polyAT, sequence)	
				mask_loc =  []
				for m in matches:
					span = m.span(0)
					for i in range(span[0], span[1]):
						mask_loc.append(i)
				mask_loc = set(mask_loc)
				seq = masked_sequence
				masked_sequence = []
				for i in range(len(seq)):
					if i in mask_loc:
						#Different character makes it easy to find when this happens
						masked_sequence.append('#')
					else:
						masked_sequence.append(sequence[i])
				masked_sequence = ''.join(masked_sequence)
		
		return is_polyAT, has_polyA, has_polyT, masked_sequence
		

	#Find regions that are AT rich above a minimum threshold
	def is_AT_rich(self, sequence):
		seq = sequence.upper()
		seq = seq.replace('-', '')
		ctr = Counter(seq)
		seqlen = len(seq)
		
		gc_content = ctr['G'] + ctr['C']
		at_content = seqlen - gc_content
		
		AT_rich_seq = gc_content >= self.AT_rich_threshold
		
		return AT_rich_seq, gc_content, at_content

	#Suffix array and longest common prefix array based approach to find seed elements of min length minimum_length
	#def find_longest_shared_subsequences(self, l, r, minimum_length = 5, desc):
	def find_longest_shared_subsequences(self, l, r):
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
				if lcp[i] >= self.exact_match_minsize:
					shared_substring_size = lcp[i]
					
					#These are not guaranteed to correspont to l or r; the lower corresponds to l, and the higher - 1 - n1 corresponds to r
					start_indices = [sa[i], sa[i + 1]]
					next_row = (min(start_indices), max(start_indices)-(n1+1), shared_substring_size)
					
					shared_substrings.append(next_row)
		
		if len(shared_substrings) > 0:
			
			shared_substrings = np.array(shared_substrings)
			
			#Unfortunately, because this is all prefixes and they are not sorted by a length or position logic, we have cleaning to do
			removal_sub = np.array([-1, -1, 1])
			
			#Sort by start location, then pattern length
			shared_substrings = shared_substrings[np.lexsort((shared_substrings[:,1], shared_substrings[:, 0],))]
			
			#Remove all strings that are a substring of another string in the list; 
			#an exact repeat is actually OK
			existing_strings = {}
			
			#First element will always be included, so we initialize with it
			this_repeat = l[shared_substrings[0, 0]:shared_substrings[0, 0] + shared_substrings[0, 2]]
			existing_strings[this_repeat] = [shared_substrings[0]]
			keepers = [0]
			
			for i in range(1, shared_substrings.shape[0]):
				#Easy check if the past row was a superstring by 1 in the correct location
				if np.all(shared_substrings[i-1] - shared_substrings[i] == removal_sub):
					keep = False
				#If easy check fails
				else:
					#Harder check to see if any kept string is a superstring in the correct location
					this_repeat = l[shared_substrings[i, 0]:shared_substrings[i, 0] + shared_substrings[i, 2]]
					keep = True
					for es in existing_strings.keys():
						if this_repeat in es:
							for loc in existing_strings[es]:
								#Get the current row
								current_row = shared_substrings[i]
								
								#Subtract the current row's starts and length from the comparison
								current_row = loc - current_row
								
								#Divide the result by the size of the repeat
								current_row = current_row / current_row[2]
								
								#If it is a substring, it will be found here
								if np.all(current_row == removal_sub):
									keep = False
									break
				if keep:
					if this_repeat not in existing_strings:
						existing_strings[this_repeat] = [shared_substrings[i]]
					else:
						existing_strings[this_repeat].append(shared_substrings[i])
					
					keepers.append(i)
					
			keepers = np.array(keepers, dtype = np.int32)
			shared_substrings = shared_substrings[keepers]

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
	
	def extend_seeds(self, left, right, exact_matches):
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
			
			if self.prevent_polyAT_extend:
				#Don't bother with checks of fewer than polyAT size, can't be confirmed as polyAT anyway
				if left_extension_size >= self.min_polyAT_len:
					left_left_is_polyAT, llpA, llpT, llcorrected = self.is_polyAT(left_extend_left_aln, clean_sequence = True)
					left_right_is_polyAT, lrpA, lrpT, lrcorrected = self.is_polyAT(left_extend_right_aln, clean_sequence = True)

					left_ok = not left_left_is_polyAT and not left_right_is_polyAT
				
					#Try to recover a non-polyAT suffix to the left extension
					if not left_ok:
						ok_size = 0
						left_mm = 0
						left_mat = 0
						
						for cl, cr in zip(reversed(llcorrected), reversed(lrcorrected)):
							if cl == '#' or cr == '#':
								break
							else:
								ok_size += 1
								if cl == cr:
									left_mat += 1
								else:
									left_mm += 1
						
						if ok_size > 0:
							llcorrected = llcorrected[-ok_size:]
							lrcorrected = lrcorrected[-ok_size:]
						else:
							llcorrected = ''
							lrcorrected = ''
							
						#Truncate leading mismatches until a match or whole string removed
						trunc = 0
						for cl, cr in zip(llcorrected, lrcorrected):
							#As soon as a match occurs, stop
							if cl == cr:
								break
							else:
								trunc += 1
						
						#Remove leading mismatches
						llcorrected = llcorrected[trunc:]
						lrcorrected = lrcorrected[trunc:]
						
						#This many mismatches were removed
						left_mm -= trunc

						left_extension_size = len(llcorrected)

						left_extend_left_aln = llcorrected
						left_extend_right_aln = lrcorrected
				
				if right_extension_size >= self.min_polyAT_len:
					right_left_is_polyAT, rlpA, rlpT, rlcorrected = self.is_polyAT(right_extend_left_aln, clean_sequence = True)
					right_right_is_polyAT, rrpA, rrpT, rrcorrected = self.is_polyAT(right_extend_right_aln, clean_sequence = True)
				
					right_ok = not right_left_is_polyAT and not right_right_is_polyAT
					
					#Try to recover a non-polyAT prefix to the right extension
					if not right_ok:
						ok_size = 0
						
						for cl, cr in zip(rlcorrected, rrcorrected):
							if cl == '#' or cr == '#':
								break
							else:
								ok_size += 1
								if cl == cr:
									right_mat += 1
								else:
									right_mm += 1
						
						rlcorrected = rlcorrected[:ok_size]
						rrcorrected = rrcorrected[:ok_size]


						#Truncate trailing mismatches
						trunc = len(rlcorrected)
						removed_mm = 0
						for cl, cr in zip(reversed(rlcorrected), reversed(rrcorrected)):
							#As soon as a match occurs, stop
							if cl == cr:
								break
							else:
								trunc -= 1
								removed_mm += 1
						
						#Remove trailing mismatches
						rlcorrected = rlcorrected[:trunc]
						rrcorrected = rrcorrected[:trunc]
						
						#This many mismatches were removed
						right_mm -= removed_mm

						right_extension_size = len(rlcorrected)

						right_extend_left_aln = rlcorrected
						right_extend_right_aln = rrcorrected
					
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

	#Find the highest scoring substring for each extension candidate and return the strings, updated extension offsets
	def score_extensions(self, extension_strings, prefer_exact = True):
		winning_candidates = []
		for candidate in extension_strings:
			
			exact_repeat = candidate[2]
			exact_score = self.score_function(len(exact_repeat), 0, 0)
			exact_mismatch = 0
			
			lq = candidate[0]
			rq = candidate[3]
			lt = candidate[1]
			rt = candidate[4]
			
			total_left_size = len(lq)
			total_right_size = len(rq)
			
			#This is an excellent point to check for polyAT
			
			#No extension found
			if total_left_size == 0 and total_right_size == 0:
				winning_candidates.append((exact_repeat, exact_repeat, exact_score, exact_mismatch, 0, 0,))
			else:
				query  = f'{lq}{exact_repeat}{rq}'
				target = f'{lt}{exact_repeat}{rt}'
			
				query = self.encode_numpy(query)
				target = self.encode_numpy(target)
				all_eq = query == target
			
				#Array of [[0/1 (false/true), start_index_in_all_eq, run_length_of_true_or_false]]
				rle_array = self.rle(all_eq)
				
				#This is indicative of some rare behavior where a non-full length exact repeat was recovered
				#There was a bug in the recovery code, but I believe I have fixed it
				if rle_array.shape[0] == 1:
					print("Should never happen now")
					#If the array is actually all matches
					'''
					if rle_array[0][0] == 1:
						score = rle_array[0][2]
						mismatch = 0
						#However large the matches are, those are our offsets
						winning_candidates.append((self.decode_numpy(query), self.decode_numpy(target), int(score), int(mismatch), total_left_size, total_right_size,))
					'''
				else:
					#This is true, false, true; only one valid substring is acceptable;
					if rle_array.shape[0] == 3:
						mismatch = np.sum(rle_array[:,2][rle_array[:,0] == 0])
						gaps = max([np.sum(query == 0), np.sum(target == 0)])
						mismatch -= gaps
						length = np.sum(rle_array[:,2])
						score = self.score_function(length, mismatch, gaps)
						
						if score > exact_score:
							#doesn't matter where the exact start is, we're grabbing the whole sequence
							winning_candidates.append((self.decode_numpy(query), self.decode_numpy(target), int(score), int(mismatch), total_left_size, total_right_size,))
						else:
							#Just return the exact match
							winning_candidates.append((exact_repeat, exact_repeat, exact_score, exact_mismatch, 0, 0,))
					else:
						
						#Which row corresponds to the start index of the exact match
						exact_start_row = rle_array[np.where(rle_array[:,1] == total_left_size)[0]][0]
						
						#print(exact_start_row)
						
						#Find subarrays that start and end with a true and internally contain no more than self.max_mismatch mismatches
						candidate_runs = self.find_valid_segments(rle_array)
						
						best_score = 0
						best_mismatch = 0
						winning_candidate = candidate_runs[0]
						offset_left = 0
						offset_right = 0
						
						for c in candidate_runs:
							my_start = c[0, 1]
							my_end = c[-1, 1]+c[-1, 2]
							
							mismatch = np.sum(c[:,2][c[:,0] == 0])
							
							gaps = max([np.sum(query[my_start:my_end] == 0), np.sum(target[my_start:my_end] == 0)])
							mismatch -= gaps
							
							length = np.sum(c[:,2])
							
							score = self.score_function(length, mismatch, gaps)
							if score > best_score:
								best_score = score
								winning_candidate = c
								best_mismatch = mismatch
						
						winning_start = winning_candidate[0, 1]
						winning_end   = winning_candidate[-1, 1] + winning_candidate[-1, 2]
						
						offset_from_exact_left  = exact_start_row[1]
						offset_from_exact_left = offset_from_exact_left - winning_start
						
						offset_from_exact_right = exact_start_row[1] + exact_start_row[2] 
						offset_from_exact_right = winning_end - offset_from_exact_right
						
						#Check if extension was successful
						if best_score > exact_score:
							winning_candidates.append((self.decode_numpy(query), self.decode_numpy(target), int(best_score), int(best_mismatch), int(offset_from_exact_left), int(offset_from_exact_right)))
						else:
							winning_candidates.append((exact_repeat, exact_repeat, exact_score, exact_mismatch, 0, 0, ))
	
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
	
	#Runner function
	def operate(self):
		shared_substrings = self.find_longest_shared_subsequences(l, r,  seq.description, minimum_length = 5)
	
	'''
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
	
	'''
	
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
									polyAT_TSD_ok = opts.poly_at_ok, 
									AT_rich_threshold = opts.polyat_threshold, 
									check_inverts = opts.check_inverts, 
									gap_penalty = opts.gap_penalty, 
									extension_penalty = opts.extension_penalty, 
									sf_score_thresh = opts.sf_score_thresh, 
									sf_mismatch_thresh = opts.sf_mismatch_thresh, 
									sf_mismatch_penalty = opts.sf_mismatch_penalty, 
									lookaround = 10,
									prevent_polyAT_extend = True,
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
			#shared_substrings = mn.find_longest_shared_subsequences(l, r, minimum_length = 5)
			shared_substrings = mn.find_longest_shared_subsequences(l, r)

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
			
				extension_loci, extension_strings = mn.extend_seeds(l, r, shared_substrings)
				
				#Extension may possibly cause sequences to overlap; filtering that is up to the user
				winners = mn.score_extensions(extension_strings)
				
				if opts.poly_at_ok:
					is_polyAT = [False] * len(winners)
				else:
					is_polyAT = [mn.is_polyAT(w[0])[0] or mn.is_polyAT(w[1])[0] for w in winners]
				
				if True:
					for sh, e, w, p in zip(shared_substrings, extension_loci, winners, is_polyAT):
						#Exclude polyAT
						if not p:
							print(sh)
							print(e)
							print(w)
							print(p)
							print('##########')
				

				
				
				
				
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
	
	