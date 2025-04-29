#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include "Ne_test.hpp"

template <typename T> 
void assert_vectors_equal(const std::vector<T>& v1, const std::vector<T>& v2) {
  // Assert that both vectors are identical
  assert(v1.size() == v2.size() && std::equal(v1.begin(), v1.end(), v2.begin()));
}

int main() {

    std::string file_path = __FILE__; 
    std::string directory_path = file_path.substr(0, file_path.find_last_of("/\\"));

    // Read in reference txt files
    NE_TEST ne_ref(directory_path);

    // Read in generated txt files 
    std::string test_data_directory_path = directory_path+"/test_data";

    NE_TEST ne_test(test_data_directory_path);

    // Check that reference parameters match test parameters
    assert(ne_ref.m_no == ne_test.m_no);    
    assert(ne_ref.m_nn == ne_test.m_nn);    
    assert_vectors_equal(ne_ref.m_Z, ne_test.m_Z);
    assert(ne_ref.m_ne == ne_test.m_ne);    
    assert(ne_ref.m_ng == ne_test.m_ng);    
    assert_vectors_equal(ne_ref.m_bf_cen, ne_test.m_bf_cen);
    assert_vectors_equal(ne_ref.m_bf_type, ne_test.m_bf_type);
    assert_vectors_equal(ne_ref.m_bf_exp, ne_test.m_bf_exp);
    assert_vectors_equal(ne_ref.m_bf_coeff, ne_test.m_bf_coeff);
    assert_vectors_equal(ne_ref.m_cusp_radii_mat, ne_test.m_cusp_radii_mat);
    assert_vectors_equal(ne_ref.m_cusp_a0_mat, ne_test.m_cusp_a0_mat);
    assert_vectors_equal(ne_ref.m_orth_orb, ne_test.m_orth_orb);
    assert_vectors_equal(ne_ref.m_proj_mat, ne_test.m_proj_mat);
    assert_vectors_equal(ne_ref.m_cusp_coeff_mat, ne_test.m_cusp_coeff_mat);
    assert_vectors_equal(ne_ref.m_n_vec, ne_test.m_n_vec);

    std::cout << "All tests passed!" << std::endl;

    return 0;
}
