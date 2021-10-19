#include "../include/ieee-754-half/half.hpp"
#include <quadmath.h>
#include <stdlib.h>
#include <stdio.h>  
#include <iostream>
#include <random>
#include <fftw3.h>

#define POISSON_MEAN 20
#define ARRAY_DIM 256
#define ARRAY_DIM_SQ ARRAY_DIM*ARRAY_DIM
#define PRINT_DIM 3
#define PRECISION 60
#define WIDTH 69

template <typename T>
void print_row(std::string name, T* data, int len )
{
  
  // if constexpr(sizeof(T) == 2*sizeof(double))
  {
    char buf[128];
    int n;
    printf("%-10s: ", name.c_str());
    if (len > 1)
    {
      for (int i=0; i<len - 1; ++i)
      {
        // should match .PRECISIONQf
        n = quadmath_snprintf (buf, sizeof buf, "%+-#*.60Qf", WIDTH, __float128(data[i]));
        printf ("%s", buf);
      }

    // should match .PRECISIONQf
      n = quadmath_snprintf (buf, sizeof buf, "%+-#*.60Qf", WIDTH, __float128(data[len]));
      printf ("%s\n", buf);
    }
    else

    {
          // should match .PRECISIONQf
      n = quadmath_snprintf (buf, sizeof buf, "%+-#*.60Qf", WIDTH, __float128(data[0]));
      printf ("%s\n", buf);
    }

  }


}

template <typename T>
void subtract_mean(T* data)
{
  T mean_val = T(0);
  int index = 0;
  for (int y=0; y<ARRAY_DIM; ++y)
  {
    for (int x=0; x<ARRAY_DIM; ++x)
    {
      mean_val += data[index];
      index++;
    }
    index += 2;
  }

  mean_val /= T(ARRAY_DIM);

  for (int i=0; i<ARRAY_DIM; ++i)
  { 
    data[i] -= mean_val;
  }
}


template <typename T>
void divide_by_constant(T* data, int len, int val)
{
  for (int i=0; i<len; ++i)
  {
    data[i] /= T(val);
  }
}



template <typename T>
void subtract_mean_row_col(T* data)
{
  T mean_val = T(0);
  int index = 0;
  for (int y = 0; y < ARRAY_DIM ; y++)
  {
    for (int x = 0; x < ARRAY_DIM; x++)
    {
      mean_val += data[y*ARRAY_DIM + x];
      index++;
    }
    index += 2;
  }
  
  mean_val = mean_val / (ARRAY_DIM_SQ);

  for (int i=0; i<ARRAY_DIM_SQ; ++i)
  {
   data[i] -= mean_val;
  } 
}

template <typename T>
void copy_from_to(T* ref, T* data, int len)
{
  for (int i=0; i<len; ++i)
  {
    data[i] = ref[i];
  }
}



template <typename Ref, typename Val>
void return_mean_std(Ref* ref, Val* data, Ref& avg, Ref& stddev)
{
  int index = 0;
  Ref relative_error;
  avg = Ref(0);
  stddev = Ref(0);

  for (int y = 0; y < ARRAY_DIM ; y++)
  {
    for (int x = 0; x < ARRAY_DIM; x++)
    {
      if (fabs(ref[index]) > 1e-6)
      {
        relative_error = fabs(__float128(data[index]) - __float128(ref[index]));
        relative_error /= __float128(ref[index]);
        avg += relative_error;
        stddev += (relative_error * relative_error);
      }

      index++;
    }
    index += 2;
  }

  avg /= ARRAY_DIM_SQ;
  stddev /= ARRAY_DIM_SQ;
  stddev -= avg*avg;
  stddev = sqrt(stddev);

  avg *= 100;
  stddev *= 100;


}

int main()
{
  using half_float::half;
  using half_float::half_cast;

  // pointers for each precision
  double* input_val;
  half* h_val;
  half* h_ref;
  float* s_val;
  float* s_ref;
  fftwf_complex* s_val_complex;
  double* d_val; 
  double* d_ref;
  fftw_complex* d_val_complex;
  __float128* q_val;
  __float128* q_ref;
  fftwq_complex* q_val_complex;



  // Setup our poisson rand variables
  std::random_device rd;
  std::mt19937 mt(rd());
  std::normal_distribution<double> dist(0.0f, 1.0f);

  // populate the integer array
  input_val = new double[ARRAY_DIM_SQ];
  for (int i = 0; i < ARRAY_DIM_SQ; i++)
  {
    input_val[i] = dist(mt);
    // make sure we all start from the same precision
    input_val[i] = half_cast<double, std::round_to_nearest>( half_cast<half, std::round_to_nearest>(input_val[i]) );
  }

  int mem_alloc = 2 * (ARRAY_DIM * (ARRAY_DIM/2+1));
  // cast to the appropriate floating point rep
  h_val = new half[mem_alloc];
  s_val = new float[mem_alloc];
  d_val = new double[mem_alloc];
  q_val = new __float128[mem_alloc];

  // keep a reference copy to test different perturbations
  h_ref = new half[mem_alloc];
  s_ref = new float[mem_alloc];
  d_ref = new double[mem_alloc];
  q_ref = new __float128[mem_alloc];

  s_val_complex = (fftwf_complex*) s_val;
  fftwf_plan plan_fwdf = NULL;
  fftwf_plan plan_bwdf = NULL;
  plan_fwdf = fftwf_plan_dft_r2c_3d(1, ARRAY_DIM,ARRAY_DIM, s_val, reinterpret_cast<fftwf_complex*>(s_val_complex), FFTW_ESTIMATE);
  plan_bwdf = fftwf_plan_dft_c2r_3d(1, ARRAY_DIM,ARRAY_DIM, reinterpret_cast<fftwf_complex*>(s_val_complex), s_val, FFTW_ESTIMATE);

  d_val_complex = (fftw_complex*) d_val;
  fftw_plan plan_fwd = NULL;
  fftw_plan plan_bwd = NULL;
  plan_fwd = fftw_plan_dft_r2c_3d(1, ARRAY_DIM,ARRAY_DIM, d_val, reinterpret_cast<fftw_complex*>(d_val_complex), FFTW_ESTIMATE);
  plan_bwd = fftw_plan_dft_c2r_3d(1, ARRAY_DIM,ARRAY_DIM, reinterpret_cast<fftw_complex*>(d_val_complex), d_val, FFTW_ESTIMATE);

  q_val_complex = (fftwq_complex*) q_val;
  fftwq_plan plan_fwdq = NULL;
  fftwq_plan plan_bwdq = NULL;
  plan_fwdq = fftwq_plan_dft_r2c_3d(1, ARRAY_DIM,ARRAY_DIM, q_val, reinterpret_cast<fftwq_complex*>(q_val_complex), FFTW_ESTIMATE);
  plan_bwdq = fftwq_plan_dft_c2r_3d(1, ARRAY_DIM,ARRAY_DIM, reinterpret_cast<fftwq_complex*>(q_val_complex), q_val, FFTW_ESTIMATE);

  for (int i = 0; i < mem_alloc; i++)
  {
    h_val[i] = half_cast<half, std::round_to_nearest>(input_val[i]); 
    h_ref[i] = h_val[i];
    s_val[i] = static_cast<float>(input_val[i]);
    s_ref[i] = s_val[i];
    d_val[i] = static_cast<double>(input_val[i]);
    d_ref[i] = d_val[i];
    q_val[i] = static_cast<__float128>(input_val[i]);
    q_ref[i] = q_val[i];
  }

  // just look at the first few values
  std::cout << "Starting from the same low-precision input: " << std::endl;
  print_row("half", h_val, PRINT_DIM);
  print_row("single", s_val, PRINT_DIM);
  print_row("double", d_val, PRINT_DIM);
  print_row("quad", q_val, PRINT_DIM);
  std::cout << std::endl;

  // subtract the mean of each array
  subtract_mean_row_col(h_val);
  subtract_mean_row_col(s_val);
  subtract_mean_row_col(d_val);
  subtract_mean_row_col(q_val);

  // just look at the first few values
  std::cout << "After subtracting the mean: " << std::endl;
  print_row("half", h_val, PRINT_DIM);
  print_row("single", s_val, PRINT_DIM);
  print_row("double", d_val, PRINT_DIM);
  print_row("quad", q_val, PRINT_DIM);
  std::cout << std::endl;

  copy_from_to(h_ref, h_val, mem_alloc);
  copy_from_to(s_ref, s_val, mem_alloc);
  copy_from_to(d_ref, d_val, mem_alloc);
  copy_from_to(q_ref, q_val, mem_alloc);

  // std::cout << "After restoring the intial values: " << std::endl;
  // print_row("half", h_val, PRINT_DIM);
  // print_row("single", s_val, PRINT_DIM);
  // print_row("double", d_val, PRINT_DIM);
  // print_row("quad", q_val, PRINT_DIM);
  // std::cout << std::endl;

  // execute round trip ffts (no half precision support on the cpu)
  fftwf_execute_dft_r2c(plan_fwdf, s_val, reinterpret_cast<fftwf_complex*>(s_val_complex));
  divide_by_constant(s_val, mem_alloc, ARRAY_DIM_SQ);
  fftwf_execute_dft_c2r(plan_bwdf, reinterpret_cast<fftwf_complex*>(s_val_complex), s_val);

  fftw_execute_dft_r2c(plan_fwd, d_val, reinterpret_cast<fftw_complex*>(d_val_complex));
  divide_by_constant(d_val, mem_alloc, ARRAY_DIM_SQ);
  fftw_execute_dft_c2r(plan_bwd, reinterpret_cast<fftw_complex*>(d_val_complex), d_val);

  fftwq_execute_dft_r2c(plan_fwdq, q_val, reinterpret_cast<fftwq_complex*>(q_val_complex));
  divide_by_constant(q_val, mem_alloc, ARRAY_DIM_SQ);
  fftwq_execute_dft_c2r(plan_bwdq, reinterpret_cast<fftwq_complex*>(q_val_complex), q_val);  

   std::cout << "After round trip fft: " << std::endl;
  // print_row("half", h_val, PRINT_DIM);
  print_row("single", s_val, PRINT_DIM);
  print_row("double", d_val, PRINT_DIM);
  print_row("quad", q_val, PRINT_DIM);
  std::cout << std::endl;   

  __float128 avg, stddev;
  return_mean_std(q_ref, q_val, avg, stddev); 
  // print_row("Quad avg", &avg, 1);
  print_row("Quad std", &stddev, 1);

  return_mean_std(q_ref, d_val, avg, stddev); 
  // print_row("double avg", &avg, 1);
  print_row("double std", &stddev, 1);

  return_mean_std(q_ref, s_val, avg, stddev); 
  // print_row("single avg", &avg, 1);
  print_row("single std", &stddev, 1);

  std::cout << std::endl;
  return_mean_std(q_val, q_val, avg, stddev); 
  // print_row("Quad avg", &avg, 1);
  print_row("Quad std", &stddev, 1);

  return_mean_std(q_val, d_val, avg, stddev); 
  // print_row("double avg", &avg, 1);
  print_row("double std", &stddev, 1);

  return_mean_std(q_val, s_val, avg, stddev); 
  // print_row("single avg", &avg, 1);
  print_row("single std", &stddev, 1);



  
  delete [] input_val;
  delete [] h_val;
  delete [] s_val;
  delete [] d_val;
  delete [] q_val;
  delete [] h_ref;
  delete [] s_ref;
  delete [] d_ref;
  delete [] q_ref;

  fftwf_destroy_plan(plan_fwdf);
  fftwf_destroy_plan(plan_bwdf);

  fftw_destroy_plan(plan_fwd);
  fftw_destroy_plan(plan_bwd);

  fftwq_destroy_plan(plan_fwdq);
  fftwq_destroy_plan(plan_bwdq);
}
