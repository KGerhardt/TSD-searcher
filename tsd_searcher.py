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
		default=1,
		help='Maximum number of total mismatches allowed under tsd_searcher method (default: 1)'
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
				sf_mismatch_thresh = 2, sf_mismatch_penalty = 1, return_best_only = True):
				
		self.method = method
		self.revcmp_table = str.maketrans('ACGTacgt', 'TGCAtgca')
		
		self.np_encoding = {'-':0, 'A':1, 'C':2, 'G':3, 'T':4}
		self.np_decoding = {0:'-', 1:'A', 2:'C', 3:'G', 4:'T'}
		
		self.min_ok_length = min_ok_length
		self.max_mismatch = max_mismatch
		self.max_consecutive_mismatches = 1
		
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
			return z, p, ia[i]

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
	
	def purge_poly_AT_seq(self, lseq, rseq):
		left_counts = np.bincount(lseq, minlength = 5)
		right_counts  = np.bincount(rseq, minlength = 5)
		
		#Correspond to sum of counts of C and G in the sequence
		left_cg = left_counts[2] + left_counts[3]
		right_cg = right_counts[2] + right_counts[3]
		
		sequence_is_ok = left_cg > self.polyAT_threshold and right_cg > self.polyAT_threshold
		
		return sequence_is_ok
	
	def extract_hit_from_df(self, df, l_enc, r_enc, gaps_l, gaps_r):
		#The place in the input strings where the shared subsequence is found
		start = df[0, 2] #First start index of a group
		end = df[-1, 2] + df[-1, 1] #last start index of a group + run length
		left_indices = l_enc[start:end]
		right_indices = r_enc[start:end]

		#Skip polyAT check; can return TSDs which are polyAT
		if self.polyAT_ok:
			is_not_poly_AT_sequence = True
		#Check sequence for A/T percentage; 
		else:
			#Check to see if the recovered sequence is a polyA or polyT or poly AT repeat; these are not TSD candidates
			is_not_poly_AT_sequence = self.purge_poly_AT_seq(left_indices, right_indices)
			
		if is_not_poly_AT_sequence:
			#Relative locations of start, end in UNGAPPED left/right input strings
			lstart = int(start - gaps_l[start])
			lend   = int(end - gaps_l[end-1]) #ends are 1-indexed in string slices, but the gap counts are still 0-indexed; left offset by 1
			rstart = int(start - gaps_r[start])
			rend   = int(end - gaps_r[end-1])
			
			tsd_length = lend - lstart
			tsd_mismatches = int(np.sum(df[df[:,0] == 0][:,1]))
			
			left = []
			right = []
			#Convert numpy ints back to characters; format mismatches accordingly
			for c1, c2 in zip([self.np_decoding[c] for c in left_indices], [self.np_decoding[c] for c in right_indices]):
				if c1 != c2:
					c1 = c1.lower()
					c2 = c2.lower()
				left.append(c1)
				right.append(c2)
			
			#Update the result to return
			left = ''.join(left)
			right = ''.join(right)
		else:
			#Default return case
			left, lstart, lend, right, rstart, rend, tsd_length, tsd_mismatches = None, None, None, None, None, None, None, None
			
		return left, lstart, lend, right, rstart, rend, tsd_length, tsd_mismatches 
	
	#Score similar sequences with sinefinder logic: 
	#TSD score = num_matches - num_mismatches; must exceed score threshold (def. 10) and must start and end with match
	def find_similar_sequences_sinefinder(self, left, right, is_forward = True):
		#Convert characters to integers and represent with numpy arrays
		#We ultimately convert back to strings at the end of this, but this makes RLE work, gap finding, etc. much easier
		left = self.encode_numpy(left)
		right = self.encode_numpy(right)
		
		gaps_left = np.cumsum(left == 0)  #counts of '-' characters for position adjustments later
		gaps_right = np.cumsum(right == 0)
		
		all_eq = left == right
		
		run_lengths, start_positions, values = self.rle(all_eq)

		winning_left = None
		winning_right = None
		winning_distance = 1_000_000
		
		winning_lstart = None
		winning_lend = None
		winning_rstart = None
		winning_rend = None
	
		current_mismatch_score = 0
		current_match_score = 0
		
		possible_candidates = []
		current_group = []
		for v, s, l in zip(values, start_positions, run_lengths):
			#Next group of matches; this will always be added
			if v:
				current_match_score += l
				current_group.append((v, l, s,))
			#Next group of mismatches; this may be added or skipped
			else:
				#the next mismatch run would exceed acceptable mismatch count
				if current_mismatch_score + (l * self.sf_pen) > self.sf_mm:
					#The current candidate is acceptable; add it to a list
					if current_match_score - current_mismatch_score >= self.sf_score and len(current_group) > 0:
						#This must end in a true because of the way the current group is constructed
						possible_candidates.append(np.array(current_group))
					
					#Reset
					current_mismatch_score = 0
					current_match_score = 0
					
					#If the next mismatch sequence couldn't be added to any segment, just proceed
					if (self.sf_pen * l) > self.sf_mm:
						current_group = []
					
					#The next mismatch segment could possibly be added to a sequence of previously observed matches
					else:
						#Pop previous true + false pairs, check if the next set of mismatches could be added
						while len(current_group) > 2:
							current_group = current_group[2:]
							for i in current_group:
								running_match = 0
								running_mismatch = 0
								#If it's a match
								if i[0]:
									#Add the number of matches
									running_match += i[1]
								else:
									#add the number of mismatches
									running_mismatch += (self.sf_pen * i[1])
							
							#If there is a remaining set of matches, 
							if running_mismatch + (self.sf_pen * l) < self.sf_mm:
								current_group.append((v, l, s,))
								current_match_score = running_match
								current_mismatch_score = running_mismatch + (self.sf_pen * l)
								break
								
						if len(current_group) == 1:
							current_mismatch_score = (self.sf_pen * l)
							current_match_score = current_group[0][1]
							current_group.append((v, l, s,))
					
				else:
					if len(current_group) > 0:
						current_group.append((v, l, s,))
						current_mismatch_score += (l * self.sf_pen)
		#Leftover group not yet added
		if len(current_group) > 0:
			#If the last element was a mismatch, pop it
			if not current_group[-1][0]:
				current_mismatch_score -= current_group[-1][1]
				current_group = current_group[:-1]
			#Add a candidate	
			if current_match_score - current_mismatch_score >= self.sf_score and len(current_group) > 0:
				#This must end in a true because of the way the current group is constructed
				possible_candidates.append(np.array(current_group))

		for df in possible_candidates:
			l, ls, le, r, rs, rend, tsdl, tsd_mm = self.extract_hit_from_df(df, left, right, gaps_left, gaps_right)
			if l is not None:
				next_candidate = (l, ls, le, r, rs, rend, tsdl, tsd_mm, is_forward,)
				self.candidates.append(next_candidate)
	
	#Score similar sequences with TSD searcher logic
	def find_similar_sequences_tsd_searcher(self, left, right, is_forward = True):
		#Convert characters to integers and represent with numpy arrays
		#We ultimately convert back to strings at the end of this, but this makes RLE work, gap finding, etc. much easier
		left = self.encode_numpy(left)
		right = self.encode_numpy(right)
		
		gaps_left = np.cumsum(left == 0)  #counts of '-' characters for position adjustments later
		gaps_right = np.cumsum(right == 0)
		
		all_eq = left == right
		
		run_lengths, start_positions, values = self.rle(all_eq)
		
		lseq = None
		rseq = None
		lstart = None
		rstart = None
		lend = None
		rend = None
		tsd_length = None
		mismatch_length = None
		
		if run_lengths.shape[0] < 3:
			'''
			Edge cases
			All matches or all mismatches, run_lengths.shape[0] = 1
			Run of mismatches - > matches or vice versa
			'''
			
			sufficient_length = False
			
			#There are either only matches or only mismatches
			if run_lengths.shape[0] == 1:
				#If it's all matches, return the strings unmodified and at full length;
				#if this is false, there is nothing to return
				if values[0]:
					if run_lengths[0] >= self.min_ok_length:
						sufficient_length = True
						tsd_length = run_lengths[0]
						lstart = 0
						rstart = 0
						lend = left.shape[0]
						rend = right.shape[0]
						
				
			#All matches followed by all mismatches or all mismatches followed by all matches
			if run_lengths.shape[0] == 2:
				#First run is matches
				if values[0]:
					if run_lengths[0] >= self.min_ok_length:
						sufficient_length = True
						tsd_length = run_lengths[0]
						lstart = 0
						rstart = 0
						lend = int(run_lengths[0])
						rend = int(run_lengths[0])

				#First run is mismatches
				else:
					if run_lengths[1] >= self.min_ok_length:
						sufficient_length = True
						tsd_length = run_lengths[1]
						lstart = int(start_positions[1])
						rstart = int(start_positions[1])
						lend = int(lstart + run_lengths[1])
						rend = int(rstart + run_lengths[1])
						
			if sufficient_length:
				mismatch_length = 0 #it always will be in these cases
				left_indices = left[lstart:lend]
				#Skip polyAT check; can return TSDs which are polyAT
				if self.polyAT_ok:
					is_not_poly_AT_sequence = True
				#Check sequence for A/T percentage; 
				else:
					#Check to see if the recovered sequence is a polyA or polyT or poly AT repeat; these are not TSD candidates
					is_not_poly_AT_sequence = self.purge_poly_AT_seq(left_indices, left_indices)
				
				if is_not_poly_AT_sequence:
					lseq = ''.join([self.np_decoding[c] for c in left_indices])
					rseq = lseq
					
					next_candidate = (lseq, lstart, lend, rseq, rstart, rend, tsd_length, tsd_mismatches, is_forward,)
					self.candidates.append(next_candidate)
					
		else:
			next_group = []					
			#Group runs of aligned sequences separated by no more than one mismatch
			for l, v, s in zip(run_lengths, values, start_positions):
				if v: #the strings have the same character over the next run_length positions
					next_group.append((v, l, s,))
				else: #The strings have a run of one or more non-matching chars
					if l == self.max_consecutive_mismatches and len(next_group) > 0: #The size of the no-match run is exactly 1
						next_group.append((v, l, s,))
					else: #The size of the no-match run is > 1 / this is a new run
						#Process the group to clean to actual putatitve TSDs
						if len(next_group) == 0:
							continue
						else:
							#Because of the way we build the above, these arrays always start and end with matches
							df = np.array(next_group)
							df_size = df.shape[0]
							#Sum the lengths of mismatch runs over all mismatches in the array
							num_mismatches = np.sum(df[df[:,0] == 0][:,1])

							#Find only the longest run of matched characters satisfying the rules
							if num_mismatches > self.max_mismatch:
								'''Rules: 
									(1) There cannot be more than max_mismatch mismatches in the string
									(2) The output must start and end with a match
									(3) Output must be at least min_ok_length characters
									(4) Return only the longest string satisfying these conditions, if any.
								'''
								#Find each consecutive sub-dataframe satisfying the above conditions
								win_start = 0
								win_length = 0
								#This logic will need updating if there's a different number of max acceptable mismatches
								check_size = 2*self.max_mismatch
								cut_size = check_size + 1
								
								for i in range(0, df_size - check_size, 2):
									#Sum of run_lengths for this dataframe = output string length; this a check to find which run of sequences is most satisfactory
									this_length = np.sum(df[i:i+cut_size, 1])
									if this_length > win_length:
										win_length = this_length
										win_start = i
								
								#Select out the winning dataframe
								df = df[win_start:win_start + cut_size, :]
							
							#The place in the input strings where the shared subsequence is found
							start = df[0, 2] #First start index of a group
							end = df[-1, 1] + df[-1, 2] #last start index of a group + run length
							
							#The longest shared subsequence passing rules still has a minimum OK size
							run_size = end - start
							if run_size >= self.min_ok_length:
								l, ls, le, r, rs, rend, tsdl, tsd_mm = self.extract_hit_from_df(df, left, right, gaps_left, gaps_right)
								if l is not None:
									next_candidate = (l, ls, le, r, rs, rend, tsdl, tsd_mm, is_forward,)
									self.candidates.append(next_candidate)

						#Reset to continue processing
						next_group = []
		
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
	
	def find_similar_sequences_lookaround(self, left, right, exact_matches, max_lookaround = 10, max_mismatch = 2):
		llen, rlen = len(left), len(right)
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
			
			ml = max_lookaround
			loop_num = 0
			num_mismatch = 0
			while continue_search:
				#No infinite loops, please
				if max_lookaround < 1:
					continue_search = False
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
					
					#Extend right along alignments until max mismatch reached
					for qc, rc in zip(list(left_aln[pattern_length:]), list(right_aln[pattern_length:])):
						extend_right += 1
						if qc != rc:
							num_mismatch += 1
							#Can't continue adding sequence
							if num_mismatch >= max_mismatch:
								continue_search = False
								break
						else:
							last_match = extend_right
							
					#Don't bother grabbing sequence if there were no matches by last mismatch
					if last_match > 0:
						right_extend_left_aln = left_aln[pattern_length:pattern_length+last_match]
						right_extend_right_aln = right_aln[pattern_length:pattern_length+last_match]
					
				ml += max_lookaround
			
			print('Original')
			print(f'{exact_repeat}')
			print('Extension')
			print(f'{exact_repeat}{right_extend_left_aln}')
			print(f'{exact_repeat}{right_extend_right_aln}')
			continue_search = True	
				
			
			
			


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
									return_best_only = opts.return_best_only)
		
		terminate = 0
		for seq in fa:
			sequence = seq.seq.upper()
			l = sequence[0:opts.left_window]
			r = sequence[-opts.right_window:]
			
			print(seq.description)
			
			#Find all longest exactly shared substrings of l and r
			#A specific substring may repeat twice in this list, but it must be in a different location to do so
			#Parameterize word size
			shared_substrings = mn.find_longest_shared_subsequences(l, r, minimum_length = 2)
						
			#I had the thought to check if a pattern starts at the same location in either string and has a longer match in the other, and to keep only the long one
			#But there are other possible useful properties of the shorter substrings (e.g., more biologically correct for some reason), so that filtering
			#Can be done elsewhere or not at all
			
			mn.find_similar_sequences_lookaround(l, r, shared_substrings)
			
			terminate += 1
			if terminate == 10:
				break
			
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
	
	