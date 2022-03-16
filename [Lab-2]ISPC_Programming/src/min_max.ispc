export void min_ispc(
        uniform int N,
        uniform int x[],
        uniform int xmin[])

{
        // assume N %  programCount = 0
        uniform int cal_min = x[0];
	uniform int temp;
	for (uniform int  i=0;  i<N;  i+=programCount)
        {
                int idx =  i +  programIndex;
		temp = reduce_min(x[idx]);
		if(cal_min > temp){
			cal_min = temp;
		}
        }
	*xmin = cal_min;

}

export void max_ispc(
        uniform int N,
        uniform int x[],
        uniform int xmax[])

{
        // assume N %  programCount = 0
	uniform int cal_max = x[0];
	uniform int temp;
        for (uniform int  i=0;  i<N;  i+=programCount)
        {
		int idx =  i +  programIndex;
		temp = reduce_max(x[idx]);
		if(cal_max < temp){
			cal_max = temp;
		}
        }
	*xmax = cal_max;
}
