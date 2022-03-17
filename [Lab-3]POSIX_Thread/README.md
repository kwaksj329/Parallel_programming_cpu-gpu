# 실행옵션

./glife <input file> <display> <nprocs> <# of generation> <width> <height>


input file: 
	sample_inputs/23334m_4505_1008
	sample_inputs/make-a_71_81
	sample_inputs/puf-qb-c3_4290_258
	sample_inputs/simple
	sample_inputs/ppt

display: 
	0 = no display (only execution time)
	1 = dump grid display
	2 = dump index display
	3 = dump grid & index display

nprocs : 
	0 = sequential version
	1 = single thread version
	2이상 = multiple threads version

 # of generation:
	number of generations

width:
	columns

height:
	rows	
