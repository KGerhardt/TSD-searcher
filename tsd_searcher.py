import sys
import os
import parasail
import numpy as np
import re
import argparse
import pyfastx


os.environ['OMP_NUM_THREADS'] = '1'

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
	parser.add_argument('--sequences', default=None,help='File of sequences in FASTA format to search for TSDs in.')
	parser.add_argument('--output', default=None, help='Output file to report results in. If none, will print to stdout')
	
	parser.add_argument('--search-mode', default='TSD', choices = ['TSD', 'TIR', 'both'], help='Select type of repeat sequence to search for.')
	
	parser.add_argument('--left-window', type=int, default=70,
		help='The first [--left-window] (default: 70) bp of each sequence in your input will searched for TSDs against the last [--right_window] bp'
	)
	parser.add_argument('--right-window', type=int, default=50,
		help='The last [--right-window] (default: 50) bp of each sequence in your input will searched for TSDs against the first [--left_window] bp'
	)
	
	parser.add_argument('--exact-match-minsize', type=int, default=5, help='Minimum size for exact NT repeats to be used as alignment seeds')
	
	parser.add_argument('--gap-penalty', type=int, default=1, help='Penalty for gaps in alignment')
	parser.add_argument('--extension-penalty', type=int, default=0, help='Penalty for extension in alignment')
	
	parser.add_argument('--lookaround', type=int, default=10, help='Search distance for alignment. Will be extended if max-mismatch has not yet been reached.')


	parser.add_argument('--min-ok-length', type=int, default=10, help='Minimum length of returned TSD/TIR sequences')
	
	parser.add_argument('--max-mismatch', type=int, default=2, help='Maximum number of mismatches allowed suring extension')
	
	parser.add_argument('--prevent-polyAT-extend', action='store_true', default=False, help='Do not permit a TSD/TIR candidate to be extended \
						with a polyA or polyT sequence'
						)
						
	parser.add_argument('--polyAT-min-length', type=int, default=5, help='Minimum length of repetitive As or Ts to be considered a polyAT sequence \
						(ex. default of 5 considers AAAAAGC to have a polyA but AAAAGC would not as there are only 4 consec. A.'
						)

	parser.add_argument('--polyAT-ok', action='store_true', default=False, help='Allow polyAT TSD/TIR. If false (default), any candidate TSD/TIR sequence \
																containing fewer than --AT-rich-threshold Gs or Cs after extension will be removed')
																
	parser.add_argument('--AT-rich-threshold', type=int, default=1, help='Minimum number of Gs or Cs to be considered acceptable. No use if --polyAT-ok is set.')
	
	parser.add_argument('--return-best-only', action='store_true', default=False, help='Return only the one best hit for TSD candidates\
						and the one best hit for TIR candidates, depending on appropriate --search-mode selection')
						
	parser.add_argument('--best-hit-approach', choices=['closest', 'high_score'], default='closest', help='Approach for selecting best hit; closest (default)\
						returns the TSD/TIR closest to (what is presumed to be) the TE candidate between them. high_score returns the candidate with the best score\
						according to --score-alg')
	
	parser.add_argument('--score-alg', choices=['kenji', 'sinefinder'], default='kenji', help='Algorithm for scoring TSDs. kenji = TSD_length - #mismatches^2,\
						sinefinder = TSD_length - #mismatches')
	
	args = parser.parse_args()
	return parser, args

#This code still needs to have a polyAT remover added sometime
#SINEfinder original TSD search behavior implemented to allow TSD searcher to recover more intact TSDs
class sinefinder_tsd_search:
	def __init__(self, exact_match_minsize = 5, max_mismatch = 1, mismatch_penalty = 1, min_ok_length = 10):
		self.exact_match_minsize = exact_match_minsize
		self.max_mismatch = max_mismatch
		self.mismatch_penalty = mismatch_penalty
		self.min_ok_length = min_ok_length

	#Rather than having this code handle forward/reverse searches, this is handled manually elsewhere in TSD searcher
	def get_best_match(self, seq1, seq2):
		"""Search in two sequences for similar subsequences. Mismatches up to
		the value of MISMATCH_TOLERANCE are accepted."""

		rseqlen = len(seq2)
		mm = ()
		offset = 0
		while 1:
			m = self.get_seed(seq1, seq2, offset)
			if not m:
				break
			em = self.extend_match(seq1, seq2, m)
			if not mm or em[3] > mm[3]:
				mm = em
				
			offset = em[0][1]
		
		'''
		if 'F' in self.cfg['TSD_ORIENTATION']:
			offset = 0
			while 1:
				m = self.get_seed(seq1, seq2, offset)
				if not m:
					break
				em = self.extend_match(seq1, seq2, m)
				if not mm or em[3] > mm[3]:
					mm = em
				offset = em[0][1]
		if 'R' in self.cfg['TSD_ORIENTATION']:
			seq3 = self._revcomp(seq2)
			offset = 0
			while 1:
				m = self.get_seed(seq1, seq3, offset)
				if not m:
					break
				em = self.extend_match(seq1, seq3, m, 'R')
				if not mm or em[3] > mm[3]:
					s = rseqlen - em[1][1];
					em[1][1] = rseqlen - em[1][0];
					em[1][0] = s;
					mm = em
				offset = em[0][1]
				
		'''
		if mm:
			#Best match wasn't long enough
			if mm[2] < self.min_ok_length:
				mm = ()
			
		return mm

	#The function of this code is implicitly done already by the SA + LCP array work in tsd searcher
	#Eventually, the logic of seedfinding should be unified and use only the SA+LCP approach
	def get_seed(self, seq1, seq2, offset=0):
		"""A window of the size of MIN_WORDSIZE shifts over sequence 1 and
		a search for all exact matches of this subsequence in sequence 2 is
		performed. All matches and their coordinates are returned. These
		serve as seed for extension."""

		seqlen = len(seq1)
		for ws in range(offset, seqlen - self.exact_match_minsize, 1):
			p = seq1[ws:min(ws + self.exact_match_minsize, seqlen)]
			i = seq2.find(p)
			if i > -1:
				return (p, ws, i)
		return None

	def extend_match(self, seq1, seq2, m):
		"""Try to extend given matches of size MIN_WORDSIZE to the right
		and to the left until MISMATCH_TOLERANCE is reached. Terminal
		mismatches will be clipped."""

		mismatches = 0
		end_mm = 0
		start_mm = 0
		is_end = 0
		
		# The seed
		m1 = [m[1], m[1] + self.exact_match_minsize - 1]
		m2 = [m[2], m[2] + self.exact_match_minsize - 1]

		# Now the extension
		while 1:
			if not is_end & 1:
				m1[1] += 1
				m2[1] += 1
				if m1[1] < len(seq1) and m2[1] < len(seq2):

					# Not at right border
					if seq1[m1[1]].upper() == seq2[m2[1]].upper():

						# Position is equal
						if end_mm:
							# The first fitting base after 1 or more mismatches
							mismatches += end_mm
							end_mm = 0

					else:

						# Position is not equal
						# add a mismatch for this direction
						end_mm += 1

						if mismatches + end_mm > self.max_mismatch:
							# This mismatch end exceeds tolerance:
							# end extension and forget about the last pos
							is_end |= 1
							m1[1] -= end_mm
							m2[1] -= end_mm
							end_mm = 0

				else:

					# Reached the right border.
					is_end |= 1

					# Clean up:
					# if it ends with mismatches, discard them.
					if end_mm:
						m1[1] -= end_mm
						m2[1] -= end_mm
						end_mm = 0
					else:
						m1[1] -= 1
						m2[1] -= 1

			if not is_end & 2:
				m1[0] -= 1
				m2[0] -= 1
				if m1[0] >= 0 and m2[0] >= 0:

					# Not at left border
					if seq1[m1[0]].upper() == seq2[m2[0]].upper():

						# Position is equal
						if start_mm:
							# The first fitting base after 1 or more mismatches
							mismatches += start_mm
							start_mm = 0

					else:

						# Position is not equal:
						# add a mismatch for this direction
						start_mm += 1

						if mismatches + start_mm > self.max_mismatch:
							# This mismatch end exceeds tolerance:
							# end this extension and forget about the last pos.
							is_end |= 2
							m1[0] += start_mm
							m2[0] += start_mm
							start_mm = 0

				else:

					# Reached the right border.
					is_end |= 2

					# Clean up:
					# if it ends with mismatches, discard them.
					if start_mm:
						m1[0] += start_mm
						m2[0] += start_mm
						start_mm = 0
					else:
						m1[0] += 1
						m2[0] += 1

			if is_end == 3 or mismatches == self.max_mismatch:
				break

		mismatches -= end_mm + start_mm
		m1[0] += start_mm
		m1[1] -= end_mm
		m2[0] += start_mm
		m2[1] -= end_mm
		length = m1[1] - m1[0] + 1
		score = length - (mismatches * self.mismatch_penalty)
		
		
		return (m1, m2, length, mismatches, score)


class alignment_tsd_tir_finder:
	def __init__(self, min_ok_length = 10, max_mismatch = 1, polyAT_TSD_ok = False, AT_rich_threshold = 1,
				check_inverts = False, gap_penalty = 1, extension_penalty = 0, lookaround = 10, prevent_polyAT_extend = False, 
				polyAT_min_length = 5, return_best_only = True, best_hit_approach = 'closest', exact_match_minsize = 5, score_alg = 'kenji'):
					
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
		
		self.check_inverts = check_inverts
		self.gap_penalty = gap_penalty
		self.ext_penalty = extension_penalty
		
		self.best = return_best_only
		self.best_hit_approach = best_hit_approach
				
		self.score_function = score_alg
		self.set_score_function()
		
		self.sinefinder_searcher = sinefinder_tsd_search(exact_match_minsize = exact_match_minsize,
														max_mismatch = max_mismatch,
														mismatch_penalty = 1)
		
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
			y = ia[1:] != ia[:-1]			   # pairwise unequal (string safe)
			i = np.append(np.where(y), n - 1)   # must include last element posi
			z = np.diff(np.append(-1, i))	   # run lengths
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
		
	#Check if a sequence is rich in certain characters
	def check_sequence_richness(self, sequence, rich_threshold = None, rich_characters = 'AT', as_percent = True):
		if rich_threshold is None:
			rich_threshold = self.AT_rich_threshold
		
		seq = sequence.upper()
		seq = seq.replace('-', '')
		ctr = Counter(seq)
		seqlen = len(seq)
		
		rich = 0
		for char in rich_characters:
			rich += ctr[char]
		
		if isinstance(rich_threshold, int):
			is_rich_sequence = (seqlen-rich) <= rich_threshold
		else:
			is_rich_sequence = (rich/seqlen) <= rich_threshold
		
		a, c, g, t = ctr['A'], ctr['C'], ctr['G'], ctr['T']
		
		if as_percent:
			a, c, g, t = a/seqlen, c/seqlen, g/seqlen, t/seqlen
		
		return is_rich_sequence, seqlen, a, c, g, t

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

	#Takes a RLE array of possible extensions and subsets to all which include:
	#exact match and <= max_mismatch mismatches/gaps
	def find_valid_segments(self, runs, anchor_index):
		segments = []
		n = runs.shape[0]
		for i in range(n):
			if runs[i][0] != 1:
				continue
			false_sum = 0
			if false_sum <= self.max_mismatch and i <= anchor_index <= i+1:
				segments.append(runs[i:i+1])
			for j in range(i + 1, n):
				if runs[j][0] == 0:
					false_sum += runs[j][2]
					if false_sum > self.max_mismatch:
						break
				else:
					if (false_sum <= self.max_mismatch or j == n - 1) and i <= anchor_index <= j+1:
						segments.append(runs[i:j+1])
		
		return segments

	#Find the highest scoring substring for each extension candidate and return the strings, updated extension offsets
	def score_extensions(self, extension_strings, prefer_exact = True):
		winning_candidates = []
		for candidate in extension_strings:
			exact_repeat = candidate[2]
			
			exact_mismatch = 0
			exact_length = len(exact_repeat)
			exact_score = self.score_function(exact_length, 0, 0)
			
			lq = candidate[0]
			rq = candidate[3]
			lt = candidate[1]
			rt = candidate[4]
			
			total_left_size = len(lq)
			total_right_size = len(rq)
			
			final_candidate = None
			
			#No extension found
			if total_left_size == 0 and total_right_size == 0:
				final_candidate = (exact_repeat, exact_repeat, exact_length, 0, 0, 0, 0, 0, 0, )
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
						left_gaps = np.sum(query == 0)
						right_gaps = np.sum(target == 0)
						gaps = max([left_gaps, right_gaps])
						mismatch -= gaps
						length = np.sum(rle_array[:,2])
						score = self.score_function(length, mismatch, gaps)
						
						
						if score > exact_score:
							left_left_offset   = total_left_size - lq.count('-')
							left_right_offset  = total_left_size - lt.count('-')
							right_left_offset  = total_right_size - rq.count('-')
							right_right_offset = total_right_size - rt.count('-')
							#doesn't matter where the exact start is, we're grabbing the whole sequence
							final_candidate = (self.decode_numpy(query), self.decode_numpy(target), int(length), int(mismatch), int(gaps), 
												left_left_offset, left_right_offset, right_left_offset, right_right_offset,)
						else:
							#Just return the exact match
							final_candidate = (exact_repeat, exact_repeat, exact_length, 0, 0, 0, 0, 0, 0, )
					else:
						
						#Which row corresponds to the start index of the exact match
						exact_start_index = np.where(rle_array[:,1] == total_left_size)[0]
						exact_start_row = rle_array[exact_start_index][0]
												
						#Find subarrays that start and end with a true and internally contain no more than self.max_mismatch mismatches
						#and must contain the exact match
						candidate_runs = self.find_valid_segments(rle_array, exact_start_index)
						
						best_score = 0
						best_mismatch = 0
						best_gaps = 0
						best_length = 0
						winning_candidate = candidate_runs[0]
						offset_left = 0
						offset_right = 0
						
						for c in candidate_runs:
							my_start = c[0, 1]
							my_end = c[-1, 1] + c[-1, 2]
							
							mismatch = np.sum(c[:,2][c[:,0] == 0])
							
							left_gaps = np.sum(query[my_start:my_end] == 0)
							right_gaps = np.sum(target[my_start:my_end] == 0)
							
							gaps = max([left_gaps, right_gaps])
							mismatch -= gaps
							
							length = np.sum(c[:,2])
							
							score = self.score_function(length, mismatch, gaps)
							if score > best_score:
								best_score = score
								best_gaps = gaps
								best_length = length
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
							query, target = self.decode_numpy(query[winning_start:winning_end]), self.decode_numpy(target[winning_start:winning_end])
							
							exact_loc = re.search(exact_repeat, query).span()
							
							query_gap_left = (query[0:exact_loc[0]]).count('-')
							target_gap_left = (target[0:exact_loc[0]]).count('-')
							query_gap_right = (query[exact_loc[1]:]).count('-')
							target_gap_right = (target[exact_loc[1]:]).count('-')
							
							offset_from_exact_left = int(offset_from_exact_left)
							offset_from_exact_right = int(offset_from_exact_right)
							
							left_left_offset   = offset_from_exact_left - query_gap_left
							left_right_offset  = offset_from_exact_left - target_gap_left
							right_left_offset  = offset_from_exact_right - query_gap_right
							right_right_offset = offset_from_exact_right - target_gap_right
						
							final_candidate = (query, target, int(best_length), int(best_mismatch), int(best_gaps), 
											   left_left_offset, left_right_offset, right_left_offset, right_right_offset,)
											
						else:
							final_candidate = (exact_repeat, exact_repeat, exact_length, 0, 0, 0, 0, 0, 0, )
			
			winning_candidates.append(final_candidate)
			
		return winning_candidates

	#If requested, return only the best matching TSD
	#Biologically, this must be the closest to the original candidate, irrespective of length
	#def get_best_hit(self, candidates, is_polyAT, shared_subs):
	def get_best_hit(self, candidates, shared_subs):
		windex = None
		if len(candidates) > 0:
			winning_score = -100_000_000
			winning_mismatch = 0
			winning_length = 0
			
			index = 0
			
			#for w, p, d in zip(candidates, is_polyAT, shared_subs):
			for w, d in zip(candidates, shared_subs):
				#sequence is not polyAT or we're not checking, in which case is_polyAT is all False
				#print(w, self.check_sequence_richness(w[0]), self.check_sequence_richness(w[1]))
			
				if self.best_hit_approach == 'high_score':
					#(query TSD, target TSD, TSD length, tsd_mismatches, tsd_gap, move_left_left, move_left_right, move_right_left, move_right_right)
					length = w[2]
					mismatch = w[3]
					gaps = w[4]
					score = length - (mismatch + gaps)
				if self.best_hit_approach == 'closest':
					#Where the left TSD candidate ends in its parent sequence; we want to maximize this
					left_end = d[0] + d[1] + w[7]
					#Where the right TSD candidate starts in its parent sequence; we want to minimize this
					right_start = d[1] - w[6]
					
					#
					score = left_end - right_start
									
				if score > winning_score:
					windex = index
					winning_score = score
				
				#Index update needs to happen even if the sequence is polyAT
				index += 1
					
			return windex
						
	#Runner function
	#Search mode indicates search TSDs only [forward_only], search inverts/TIRs only [invert_only] or both [both]
	def operate(self, left_sequence, right_sequence, is_TIR = False):
		if is_TIR:
			right_sequence = self.revcomp(right_sequence)
		
		final_tsds = []
		shared_substrings = self.find_longest_shared_subsequences(left_sequence, right_sequence)
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
			#Currently extension loci goes unused
			extension_loci, extension_strings = self.extend_seeds(left_sequence, right_sequence, shared_substrings)
			
			#Score extensions; for each successful extension, truncate to the highest scoring run with <= self.max_mismatch
			#Returns all exact matches and their extensions with score, mismatch count, and 
			winners = self.score_extensions(extension_strings)
			if self.polyAT_TSD_ok:
				is_polyAT = [False] * len(winners)
			else:
				#Check if the sequence contains a polyA or polyT sequence
				#is_polyAT = [mn.is_polyAT(w[0], clean_sequence = False)[0] or 
				#			 mn.is_polyAT(w[1], clean_sequence = False)[0] for w in winners]
				
				#Check if the sequence is all A/T
				is_polyAT = [self.check_sequence_richness(w[0],)[0] or self.check_sequence_richness(w[1])[0] for w in winners]

			is_not_polyAT = np.logical_not(np.array(is_polyAT))
			acceptable_length = np.zeros(shape = is_not_polyAT.shape, dtype = np.bool_)
			for i in range(0, len(winners)):
				if winners[i][2] >= self.min_ok_length:
					acceptable_length[i] = True
			
			retained_sequences = np.logical_and(is_not_polyAT, acceptable_length)
			
			#print(is_not_polyAT)
			#print(acceptable_length)
			
			#Ensure 2-d retention of shared substrings
			shared_substrings = shared_substrings[retained_sequences, :]
			winners = [winners[i] for i in np.where(retained_sequences)[0]]
				
			#Filter to best hit
			if self.best:
				#best_index = self.get_best_hit(winners, is_polyAT, shared_substrings)
				best_index = self.get_best_hit(winners, shared_substrings)
				if best_index is not None:
					winners = [winners[best_index]]
					shared_substrings = shared_substrings[best_index, None]
				else:
					winners = None
					shared_substrings = None

			if winners is not None:
				#Winners has form
				#(query TSD, target TSD, TSD length, tsd_mismatches, tsd_gap, move_left_left, move_left_right, move_right_left, move_right_right)
				for original_indices, updates in zip(shared_substrings, winners):
					left_string_start = int(original_indices[0] - updates[5])
					right_string_start = int(original_indices[1] - updates[6])
					left_string_end = int(original_indices[0] + original_indices[2] + updates[7])
					right_string_end = int(original_indices[1] + original_indices[2] + updates[8])
					
					if is_TIR:
						right_sequence_length = len(right_sequence)
						reverted_right = self.revcomp(updates[1])
						
						#Forward orientation substring coordinates
						right_string_start_fo = right_sequence_length - right_string_end
						right_string_end_fo = right_sequence_length - right_string_start
						
						tsd = (updates[0], reverted_right, left_string_start, left_string_end, right_string_start_fo, right_string_end_fo, updates[2], updates[3]+updates[4], )
					
					else:
						tsd = (updates[0], updates[1], left_string_start, left_string_end, right_string_start, right_string_end, updates[2], updates[3]+updates[4], )
					
					#Print checks
					#if '-' in updates[0] or '-' in updates[1]:
					#	print(seq.description)
					#	print(f'{updates[0]} {updates[1]} {left_string_start}:{left_string_end} {right_string_start}:{right_string_end} tsd_length:{updates[2]} tsd_mismatches:{updates[3]+updates[4]}')
					#	print('')
					#print(f'left_seq	 {tsd[0]}')
					#print(f'right_seq	{tsd[1]}')
					#print(f'left select  {l[tsd[2]:tsd[3]]}')
					#print(f'right select {r[tsd[4]:tsd[5]]}')
					
					final_tsds.append(tsd)
			
		if len(final_tsds) == 0:
			#Fallback to SINEfinder TSD search
			match = self.sinefinder_searcher.get_best_match(left_sequence, right_sequence)
			if match:
				left_indices, right_indices, length, mismatches, score = match
				left_string_start, left_string_end = left_indices
				right_string_start, right_string_end = right_indices
				left_tsd = left_sequence[left_string_start:left_string_end]
				right_tsd = right_sequence[right_string_start:right_string_end]
				if is_TIR:
					right_sequence_length = len(right_sequence)
					right_tsd = self.revcomp(right_tsd)
					right_string_start_fo = right_sequence_length - right_string_end
					right_string_end_fo = right_sequence_length - right_string_start
					right_string_start = right_string_start_fo
					right_string_end = right_string_end_fo
				
				sinefinder_tsd = (left_tsd, right_tsd, left_string_start, left_string_end, right_string_start, right_string_end, length, mismatches,)
				final_tsds = [sinefinder_tsd]
			else:
				final_tsds= None
			
			
			#left_tsd, right_tsd, left_loc_start, left_loc_end, right_loc_start, right_loc_end, 
			#tsd = (updates[0], updates[1], left_string_start, left_string_end, right_string_start, right_string_end, updates[2], updates[3]+updates[4], )


		return final_tsds
		
	
def main():
	parser, opts = options()
	
	#the OMP parallelization is good for suffix arrays of long strings, usually inappropriate to TSD searching
	
	
	if opts.sequences is None:
		print('You must supply a set of sequences to search for TSDs using --sequences')
	else:
		fa = pyfastx.Fasta(opts.sequences)
		
		output = opts.output
		
		search_TSD = opts.search_mode == 'TSD' or opts.search_mode == "both"
		search_TIR = opts.search_mode == 'TIR' or opts.search_mode == "both"
		
		mn = alignment_tsd_tir_finder(min_ok_length = opts.min_ok_length, 
								max_mismatch = opts.max_mismatch, 
								polyAT_TSD_ok = opts.polyAT_ok, 
								AT_rich_threshold = opts.AT_rich_threshold, 
								gap_penalty = opts.gap_penalty, 
								extension_penalty = opts.extension_penalty, 
								lookaround = opts.lookaround,
								prevent_polyAT_extend = opts.prevent_polyAT_extend, 
								polyAT_min_length = opts.polyAT_min_length, 
								return_best_only = opts.return_best_only, 
								best_hit_approach = opts.best_hit_approach, 
								exact_match_minsize = opts.exact_match_minsize, 
								score_alg = opts.score_alg)
		
		if output is not None:
			out = open(output, 'w')
		else:
			out = sys.stdout
		
		for seq in fa:
			final_tsds = []
			sequence = seq.seq.upper()
			l = sequence[0:opts.left_window]
			r = sequence[-opts.right_window:]
			
			if search_TSD:
				tsds = mn.operate(l, r, is_TIR = False)
			else:
				tsds = None
			
			if tsds is not None:
				for t in tsds:
					print(seq.description, *t, 'TSD', sep = '\t', file = out)

			if search_TIR:
				tirs = mn.operate(l, r, is_TIR = True)
			else:
				tirs = None
				
			if tirs is not None:
				for t in tirs:
					print(seq.description, *t, 'TIR', sep = '\t', file = out)
					
		if output is not None:
			out.close()

if __name__ == '__main__':
	main()
	
	