#include "glife.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

int gameOfLife(int argc, char *argv[]);
void singleThread(int, int, int);
void* workerThread(void *);
int nprocs;
GameOfLifeGrid* g_GameOfLifeGrid;

pthread_barrier_t barrier;   

typedef struct{
	int from_rows;
	int to_rows;
	int cols;
}t_args;

uint64_t dtime_usec(uint64_t start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

GameOfLifeGrid::GameOfLifeGrid(int rows, int cols, int gen)
{
  m_Generations = gen;
  m_Rows = rows;
  m_Cols = cols;

  m_Grid = (int**)malloc(sizeof(int*) * rows);
  if (m_Grid == NULL) 
    cout << "1 Memory allocation error " << endl;

  m_Temp = (int**)malloc(sizeof(int*) * rows);
  if (m_Temp == NULL) 
    cout << "2 Memory allocation error " << endl;

  m_Grid[0] = (int*)malloc(sizeof(int) * (cols*rows));
  if (m_Grid[0] == NULL) 
    cout << "3 Memory allocation error " << endl;

  m_Temp[0] = (int*)malloc(sizeof(int) * (cols*rows));	
  if (m_Temp[0] == NULL) 
    cout << "4 Memory allocation error " << endl;

  for (int i = 1; i < rows; i++) {
    m_Grid[i] = m_Grid[i-1] + cols;
    m_Temp[i] = m_Temp[i-1] + cols;
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m_Grid[i][j] = m_Temp[i][j] = 0;
    }
  }
}

// Entry point
int main(int argc, char* argv[])
{
  if (argc != 7) {
    cout <<"Usage: " << argv[0] << " <input file> <display> <nprocs>"
           " <# of generation> <width> <height>" << endl;
    cout <<"\n\tnprocs = 0: Running sequentiallyU" << endl;
    cout <<"\tnprocs =1: Running on a single thread" << endl;
    cout <<"\tnprocs >1: Running on multiple threads" << endl;
    cout <<"\tdisplay = 1: Dump results" << endl;
    return 1;
  }

  return gameOfLife(argc, argv);
}

int gameOfLife(int argc, char* argv[])
{
  int cols, rows, gen;
  ifstream inputFile;
  int input_row, input_col, display;
  uint64_t difft;
  pthread_t *threadID;

  inputFile.open(argv[1], ifstream::in);

  if (inputFile.is_open() == false) {
    cout << "The "<< argv[1] << " file can not be opend" << endl;
    return 1;
  }

  display = atoi(argv[2]);
  nprocs = atoi(argv[3]);
  gen = atoi(argv[4]);
  cols = atoi(argv[5]);
  rows = atoi(argv[6]);

  g_GameOfLifeGrid = new GameOfLifeGrid(rows, cols, gen);

  while (inputFile.good()) {
    inputFile >> input_row >> input_col;
    if (input_row >= rows || input_col >= cols) {
      cout << "Invalid grid number" << endl;
      return 1;
    } else
      g_GameOfLifeGrid->setCell(input_row, input_col);
  }

  // Start measuring execution time
  difft = dtime_usec(0);

  // TODO: YOU NEED TO IMPLMENT THE SINGLE THREAD and PTHREAD
  if (nprocs == 0) {
    // Running with your sequential version
        
	while(g_GameOfLifeGrid->getGens() > 0){
		
		int neighbor = 0;
		for(int i = 0 ; i < rows ; i++){
			for(int j = 0 ; j < cols ; j++){
				
				neighbor = g_GameOfLifeGrid->getNumOfNeighbors( i, j );
				
				if(g_GameOfLifeGrid->isLive(i, j)){
					if(neighbor == 2 || neighbor ==3){
						g_GameOfLifeGrid->live(i, j);
					}else{
						g_GameOfLifeGrid->dead(i, j);
					}
				}else{
					if(neighbor == 3){
						g_GameOfLifeGrid->live(i, j);
					}else{
						g_GameOfLifeGrid->dead(i, j);
					}
				
				}
                	}
        	}

		g_GameOfLifeGrid->next();

	}

  } else if(nprocs >= 1){ 
    // Running single / multiple threads (pthread)
    
	  int t_id;
	  pthread_t pthread[nprocs];
	  pthread_barrier_init(&barrier, NULL, nprocs);

	  t_args* args[nprocs];
	  for(int i = 0 ; i < nprocs ; i++){
	  	args[i] = (t_args*)malloc(sizeof(t_args));
		args[i]->from_rows = i * (rows/nprocs);
		if( i+1 == nprocs)
			args[i]->to_rows = rows;
		else
			args[i]->to_rows = (i+1) * (rows/nprocs);
		args[i]->cols = cols;

		t_id = pthread_create(&pthread[i], NULL, workerThread, (void*)args[i]);

		if(t_id < 0){
			printf("Multiple thread create error : %d\n", t_id);
			exit(0);
		}

	  }

	  for(int i = 0 ; i < nprocs ; i++){

		t_id = pthread_join(pthread[i], NULL);

                if(t_id){
                        printf("Error; return code from pthread_join() is %d\n", t_id);
                        exit(-1);
                }

                //printf("Completed join with thread: %d\n", (int)pthread[i]);
	  
	  }

  }

  difft = dtime_usec(difft);

  // Print indices only for running on CPU(host).
  if (display == 1) {
    g_GameOfLifeGrid->dump();
  }else if(display == 2){
    g_GameOfLifeGrid->dumpIndex();
  }else if(display == 3){
    g_GameOfLifeGrid->dump();
    g_GameOfLifeGrid->dumpIndex();
  }

  if (nprocs == 0) {
    // Sequential version
    cout << "Execution time(seconds): " << difft/(float)USECPSEC << endl;
  } else if (nprocs >= 1) {
    // Single or multi-thread execution time 
    cout << "Execution time(seconds): " << difft/(float)USECPSEC << endl;
  }
  inputFile.close();
  cout << "Program end... " << endl;
  return 0;
}


// TODO: YOU NEED TO IMPLMENT PTHREAD
void* workerThread(void *arg)
{
	t_args* args = (t_args *)arg;

        while(g_GameOfLifeGrid->getGens() > 0){

                int neighbor = 0;

                for(int i = args->from_rows ; i < args->to_rows ; i++){
                        for(int j = 0 ; j < args->cols ; j++){

                                neighbor = g_GameOfLifeGrid->getNumOfNeighbors( i, j );

                                if(g_GameOfLifeGrid->isLive(i, j)){
                                        if(neighbor == 2 || neighbor ==3){
                                                g_GameOfLifeGrid->live(i, j);
                                        }else{
                                                g_GameOfLifeGrid->dead(i, j);
                                        }
                                }else{
                                        if(neighbor == 3){
                                                g_GameOfLifeGrid->live(i, j);
                                        }else{
                                                g_GameOfLifeGrid->dead(i, j);
                                        }

                                }
                        }
                }

                int lastwait = pthread_barrier_wait(&barrier);
                if(lastwait == PTHREAD_BARRIER_SERIAL_THREAD){
                        g_GameOfLifeGrid->decGen();
                }
                g_GameOfLifeGrid->next(args->from_rows, args->to_rows);
        	pthread_barrier_wait(&barrier);
	}

}

// HINT: YOU MAY NEED TO FILL OUT BELOW FUNCTIONS OR CREATE NEW FUNCTIONS
void GameOfLifeGrid::next(const int from, const int to)
{
	for(int i = from ; i < to ; i++){
		for(int j = 0 ; j < m_Cols ; j++ ){
			m_Grid[i][j] = m_Temp[i][j];
		}
	}

}

void GameOfLifeGrid::next()
{
	for(int i = 0 ; i < m_Rows ; i++){
		for(int j = 0 ; j < m_Cols ; j++){
			m_Grid[i][j] = m_Temp[i][j];
		}
	}

	decGen();
}


// TODO: YOU MAY NEED TO IMPLMENT IT TO GET NUMBER OF NEIGHBORS 
int GameOfLifeGrid::getNumOfNeighbors(int rows, int cols)
{
	int neighbor = 0;
	
	if(rows - 1 >= 0){
		if(cols-1 >= 0 && isLive(rows-1, cols-1))	neighbor++;

		if(cols+1 < m_Cols && isLive(rows-1, cols+1))	neighbor++;
		
		if(isLive(rows-1, cols))	neighbor++;
	}
	
	if(cols-1 >= 0 && isLive(rows, cols-1))	neighbor++;
	if(cols+1 < m_Cols && isLive(rows, cols+1))	neighbor++;

	if(rows+1 < m_Rows){
		if( cols-1 >= 0 && isLive(rows+1, cols-1))	neighbor++;

		if(cols+1 < m_Cols && isLive(rows+1, cols+1))	neighbor++;
		
		if(isLive(rows+1, cols))	neighbor++;
	}

	return neighbor;
	
}

void GameOfLifeGrid::dump() 
{
  cout << "===============================" << endl;

  for (int i = 0; i < m_Rows; i++) {
    cout << "[" << i << "] ";
    for (int j = 0; j < m_Cols; j++) {
      if (m_Grid[i][j] == 1)
        cout << "*";
      else
        cout << "o";
    }
    cout << endl;
  }
  cout << "===============================\n" << endl;
}

void GameOfLifeGrid::dumpIndex()
{
  cout << ":: Dump Row Column indices" << endl;
  for (int i=0; i < m_Rows; i++) {
    for (int j=0; j < m_Cols; j++) {
      if (m_Grid[i][j]) cout << i << " " << j << endl;
    }
  }
}
