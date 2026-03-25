## Testing logic variables
#Rates to test
rates = [1.5e6, 3e6, 4.5e6, 6e6, 7.5e6, 9e6]
labeled_rate_dict = dict(zip(['A','B','C','D','E','F'], rates))

sequence_freq_keys = {0:['A', 'B', 'F', 'C', 'E', 'D'],
                      1:['B', 'C', 'A', 'D', 'F', 'E'],
                      2:['C', 'D', 'B', 'E', 'A', 'F'],
                      3:['D', 'E', 'C', 'F', 'B', 'A'],
                      4:['E', 'F', 'D', 'A', 'C', 'B'],
                      5:['F', 'A', 'E', 'B', 'D', 'C']}
sequence_dict = {seq:[labeled_rate_dict[key] for key in sequence_freq_keys[seq]]for seq in sequence_freq_keys.keys()}

#Locations and replicates in a dict
locations = dict(zip(list(range(7)), ['RL', 'CL', 'FL', 'FR', 'CR', 'RR','XX']))
replicate_loc_key_dict = {0:[0, 6, 4, 3, 2, 5, 1],
                          1:[5, 0, 3, 6, 4, 1, 2],
                          2:[4, 3, 2, 1, 5, 0, 6],
                          3:[3, 2, 1, 5, 0, 6, 4],
                          4:[2, 5, 6, 4, 1, 3, 0],
                          5:[6, 1, 0, 2, 3, 4, 5],
                          6:[1, 4, 5, 0, 6, 2, 3]}
replicate_dict = {rep:[locations[key] for key in replicate_loc_key_dict[rep]] for rep in replicate_loc_key_dict.keys()}