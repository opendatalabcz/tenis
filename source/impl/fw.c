void FW(float * dis, int * next, int size, int INF)
{
  for(int k=0; k<size; k+=1)
  {
    for (int i=0; i<size; i+=1)
     {
        for(int j=0; j<size; j+=1)
        {
          
            if (dis[i*size+k] != INF && dis[k*size+j] != INF)
            {
                 if (dis[i*size+j] > (dis[i*size+k]+dis[k*size+j]))
                 {
                   dis[i*size+j] = (dis[i*size+k]+dis[k*size+j]);
                   next[i*size+j] = next[i*size+k];
                 }
            }
        }
      }
   }
}

