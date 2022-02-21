#include <iostream>
#include <windows.h>
#include <immintrin.h>
using namespace std;
#define CELL_SIZE 12
#define MATRIX_SIZE 100
#define SIZE  CELL_SIZE*MATRIX_SIZE
unsigned long long multiplication(double** first_matrix, double** second_matrix, double** result_matrix)
{
    unsigned long long execution_time1 = GetTickCount64();
    for (int i = 0; i < SIZE; ++i)
    {
        // ���������� ������ ��������������� �������
        double* result = result_matrix[i];
        for (int k = 0; k < SIZE; ++k)
        {
            // ��������� ������� ������ �������
            const double* second = second_matrix[k];
            // ��������� ������� ������ �������
            double first = first_matrix[i][k];
            // ������������� ��������� ������������
#pragma loop(no_vector)
            for (int j = 0; j < SIZE; ++j)
                // ������������. �������
                result[j] += first * second[j];
        }
    }
    unsigned long long execution_time2 = GetTickCount64();
    return (execution_time2 - execution_time1);
}
unsigned long long vectorise_multiplication(double** first_matrix, double** second_matrix, double** result_matrix)
{
    unsigned long long execution_time1 = GetTickCount64();
    for (int i = 0; i < SIZE; ++i)
    {
        // ���������� ������ ��������������� �������
        double* result = result_matrix[i];
        // ���������� �� �������� ������
        for (int k = 0; k < SIZE; ++k)
        {
            // ��������� ������� ������ �������
            const double* second = second_matrix[k];
            // ��������� ������� ������ �������
            double first = first_matrix[i][k];
            for (int j = 0; j < SIZE; ++j)
                // ������������. �������
                result[j] += first * second[j];
        }
    }
    unsigned long long execution_time2 = GetTickCount64();
    return (execution_time2 - execution_time1);
}
unsigned long long intrinsics_SSE_vectorise_multiplication(double** first_matrix, double** second_matrix, double** result_matrix)
{
    unsigned long long execution_time1 = GetTickCount64();
    __m128d result;
    __m128d first;
    __m128d second;
    for (int i = 0; i < SIZE; ++i)
    {
        // ���������� ������ ��������������� �������
        double* m_result = result_matrix[i];
        for (int k = 0; k < SIZE; k++)
        {
            // ��������� ������� ������ �������
            const double* m_second = second_matrix[k];
            // ��������� ������� ������ �������
            first = _mm_set1_pd(first_matrix[i][k]);
            for (int j = 0; j < SIZE; j+=4)
            {
                result = _mm_load_pd(m_result + j);
                second = _mm_load_pd(m_second + j);
                result = _mm_add_pd(result, _mm_mul_pd(first, second));
                _mm_store_pd(m_result + j, result);

                result = _mm_load_pd(m_result + j+2);
                second = _mm_load_pd(m_second + j+2);
                result = _mm_add_pd(result, _mm_mul_pd(first, second));
                _mm_store_pd(m_result + j+2, result);
            }
        }
    }
    unsigned long long execution_time2 = GetTickCount64();
    return (execution_time2 - execution_time1);
}
bool compare_results(double** first_result, double** second_result, double** third_result)
{
    // �-��� ��������� ������ �� ���������
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (first_result[i][j] != second_result[i][j] ||
                first_result[i][j] != third_result[i][j] ||
                third_result[i][j] != second_result[i][j])
                return false;
    return true;
}
int main()
{
    double** first_matrix;
    double** second_matrix;
    double** result_matrix;
    double** first_result;
    double** second_result;
    double** third_result;
    // ��������� ������
    cout << "MEMPORY ALLOCATION...\n";
    first_matrix = (double**)calloc(SIZE, sizeof(double*));
    second_matrix = (double**)calloc(SIZE, sizeof(double*));
    first_result = (double**)calloc(SIZE, sizeof(double*));
    second_result = (double**)calloc(SIZE, sizeof(double*));
    third_result = (double**)calloc(SIZE, sizeof(double*));
    for (int i = 0; i < MATRIX_SIZE * CELL_SIZE; i++)
    {
        first_matrix[i] = (double*)calloc(SIZE, sizeof(double));
        second_matrix[i] = (double*)calloc(SIZE, sizeof(double));
        first_result[i] = (double*)calloc(SIZE, sizeof(double));
        second_result[i] = (double*)calloc(SIZE, sizeof(double));
        third_result[i] = (double*)calloc(SIZE, sizeof(double));
    }
    // ������������� ������
    cout << "MEMORY INITIALIZATION...\n";
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            first_matrix[i][j] = rand();
            second_matrix[i][j] = rand() ;
            first_result[i][j] = 0;
            second_result[i][j] = 0;
            third_result[i][j] = 0;
        }
    // ��������� ��� ������������
    cout << "START MULTIPLICATION...\n\n";
    cout << "WITHOUT Vectorise: " << multiplication(first_matrix, second_matrix, first_result) <<" ms\n";
    // ��������� � �������������� �������������
    cout << "WITH Vectorise: " << vectorise_multiplication(first_matrix, second_matrix, second_result) << " ms\n";
    // ��������� � ������ ������������� SSE
    cout << "WITH HAND SSE Vectorise: " << intrinsics_SSE_vectorise_multiplication(first_matrix, second_matrix, third_result) << " ms\n\n";
    // �������� ����������� ��������� �� ���������
    cout << "START COMPARING...\n";
    compare_results(first_result, second_result, third_result) == true ? cout << "MATRIX ARE EQUAL\n" : cout << "MATRIX ARE not EQUAL\n";
    cout << "END WORKING...\n";
    // ������������ ������s
    for (int i = 0; i < SIZE; i++)
    {
        free(first_matrix[i]);
        free(second_matrix[i]);
        free(first_result[i]);
        free(second_result[i]);
        free(third_result[i]);
    }
    free(first_matrix);
    free(second_matrix);
    free(first_result);
    free(second_result);
    free(third_result);
    return 0;
}
