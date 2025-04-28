import pytest
import beyonce.validate as validate
import numpy as np
from beyonce.shallot.errors import InvalidDimensionsError, InvalidShapeError
from beyonce.shallot.errors import InvalidBoundsError, ValueOutsideRangeError

NAME = "parameter"



###############################################################################
################################# boolean() ###################################
###############################################################################


def test_boolean_invalid_number() -> None:
    parameter = np.random.randint(0, 1000)
    
    with pytest.raises(TypeError) as ERROR:
        validate.boolean(parameter, NAME)

    assert str(ERROR.value) == f"{NAME} is not of type boolean"


def test_boolean_invalid_boolean_collection() -> None:
    parameter = [False, True]
    
    with pytest.raises(TypeError) as ERROR:
        validate.boolean(parameter, NAME)

    assert str(ERROR.value) == f"{NAME} is not of type boolean"


def test_boolean_valid() -> None:
    parameter = True
    parameter_out = validate.boolean(parameter, NAME)
    assert parameter_out == parameter



###############################################################################
################################## string() ###################################
###############################################################################


def test_string_invalid_number() -> None:
    parameter = np.random.randint(0, 1000)
    
    with pytest.raises(TypeError) as ERROR:
        validate.string(parameter, NAME)

    assert str(ERROR.value) == f"{NAME} is not of type string"


def test_string_invalid_string_collection() -> None:
    parameter = ["test_string"]
    
    with pytest.raises(TypeError) as ERROR:
        validate.string(parameter, NAME)

    assert str(ERROR.value) == f"{NAME} is not of type string"


def test_string_valid() -> None:
    parameter = "test"
    validate.string(parameter, NAME)
    assert str(parameter) == parameter



###############################################################################
################################## number() ###################################
###############################################################################

""" NUMBER TYPE """
""" ----------- """


def test_number_type_integer_check_false() -> None:
    parameter = np.random.randint(0, 1000) 
    validate.number(parameter, NAME, check_integer=False)
    assert int(parameter) == parameter


def test_number_type_float_check_false() -> None:
    parameter = np.random.random()   
    validate.number(parameter, NAME, check_integer=False)
    assert float(parameter) == parameter


def test_number_type_integer_check_true() -> None:
    parameter = np.random.randint(0, 1000) 
    validate.number(parameter, NAME, check_integer=True)
    assert int(parameter) == parameter


def test_number_type_float_check_true() -> None:
    parameter = np.random.random()  
    
    with pytest.raises(TypeError) as ERROR: 
        validate.number(parameter, NAME, check_integer=True)
    
    assert str(ERROR.value) == f"{NAME} is not of type integer"


def test_number_type_any_check_invalid() -> None:
    parameter = np.random.random()

    with pytest.raises(TypeError) as ERROR:
        validate.number(parameter, NAME, "invalid")
    
    assert str(ERROR.value) == "check_integer is not of type boolean"


def test_number_type_any_check_invalid_like() -> None:
    parameter = np.random.random()

    with pytest.raises(TypeError) as ERROR:
        validate.number(parameter, NAME, 0)
    
    assert str(ERROR.value) == "check_integer is not of type boolean"

""" NUMBER BOUNDS """
""" ------------- """


def test_number_bounds_invalid_lower() -> None:
    parameter = 1
    with pytest.raises(TypeError) as ERROR:
        validate.number(parameter, NAME, lower_bound="invalid")

    assert str(ERROR.value) == "lower_bound is not of type 'number'"


def test_number_bounds_invalid_upper() -> None:
    parameter = 1
    with pytest.raises(TypeError) as ERROR:
        validate.number(parameter, NAME, upper_bound="invalid")

    assert str(ERROR.value) == "upper_bound is not of type 'number'"


def test_number_bounds_invalid_exclusive() -> None:
    parameter = 1
    with pytest.raises(TypeError) as ERROR:
        validate.number(parameter, NAME, exclusive="invalid")

    assert str(ERROR.value) == "exclusive is not of type boolean"


def test_number_bounds_lower_greater_than_upper() -> None:
    parameter = 1
    lower_bound = 0
    upper_bound = -1
    with pytest.raises(InvalidBoundsError) as ERROR:
        validate.number(parameter, NAME, lower_bound=lower_bound,
            upper_bound=upper_bound)

    assert str(ERROR.value) == (
        f"The upper_bound argument ({upper_bound:.4f}) must be greater than "
        f"the lower_bound argument ({lower_bound:.4f})."
    )


def test_number_bounds_value_low_exclusive_false() -> None:
    parameter = 1
    lower_bound = 2
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.number(parameter, NAME, lower_bound=lower_bound)
    
    assert str(ERROR.value) == (f"The {NAME} argument ({parameter:.4f}) must "
        f"be between or equal to {lower_bound:.4f} and inf.")


def test_number_bounds_value_low_exclusive_true() -> None:
    parameter = 1
    lower_bound = 1
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.number(parameter, NAME, lower_bound=lower_bound, 
            exclusive=True)
    
    assert str(ERROR.value) == (f"The {NAME} argument ({parameter:.4f}) must "
        f"be between {lower_bound:.4f} and inf.")


def test_number_bounds_value_high_exclusive_false() -> None:
    parameter = 1
    upper_bound = 0
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.number(parameter, NAME,upper_bound=upper_bound)
    
    assert str(ERROR.value) == (f"The {NAME} argument ({parameter:.4f}) must "
        f"be between or equal to -inf and {upper_bound:.4f}.")


def test_number_bounds_value_high_exclusive_true() -> None:
    parameter = 1
    upper_bound = 1
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.number(parameter, NAME, upper_bound=upper_bound, 
            exclusive=True)
    
    assert str(ERROR.value) == (f"The {NAME} argument ({parameter:.4f}) must "
        f"be between -inf and {upper_bound:.4f}.")


def test_number_bounds_valid_defaults() -> None:
    parameter = 1
    
    parameter = validate.number(parameter, NAME)
    
    assert True


def test_number_bounds_valid_lower_exclusive_false() -> None:
    parameter = 1
    lower_bound = 1
    
    parameter = validate.number(parameter, NAME, lower_bound=lower_bound)
    
    assert True


def test_number_bounds_valid_upper_exclusive_false() -> None:
    parameter = 1
    upper_bound = 1

    parameter = validate.number(parameter, NAME, upper_bound=upper_bound)

    assert True


def test_number_bounds_equal_exclusive_false() -> None:
    parameter = 0
    lower_bound = 0
    upper_bound = 0

    parameter = validate.number(parameter, NAME, lower_bound=lower_bound, 
        upper_bound=upper_bound, exclusive=False)

    assert True



###############################################################################
################################## array() ####################################
###############################################################################

""" ARRAY TYPE """
""" ---------- """


def test_array_invalid() -> None:
    parameter = np.random.randint(0, 1000)
    
    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME)

    assert str(ERROR.value) == f"{NAME} is not of type np.ndarray"


def test_array_invalid_collection() -> None:
    parameter = [np.ones(2), np.zeros(1)]
    
    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME)

    assert str(ERROR.value) == f"{NAME} is not of type np.ndarray"


def test_array_valid() -> None:
    parameter = np.zeros(2)
    
    parameter = validate.array(parameter, NAME)

    assert True

""" ARRAY BOUNDS """
""" ------------ """


def test_array_bounds_invalid_lower() -> None:
    parameter = np.ones(3)
    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME, lower_bound="invalid")

    assert str(ERROR.value) == "lower_bound is not of type 'number'"


def test_array_bounds_invalid_upper() -> None:
    parameter = np.ones(3)
    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME, upper_bound="invalid")

    assert str(ERROR.value) == "upper_bound is not of type 'number'"


def test_array_bounds_invalid_exclusive() -> None:
    parameter = np.ones(3)
    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME, lower_bound=0, exclusive="invalid")

    assert str(ERROR.value) == "exclusive is not of type boolean"


def test_array_bounds_lower_greater_than_upper() -> None:
    parameter = np.ones(3)
    lower_bound = 0
    upper_bound = -1
    with pytest.raises(InvalidBoundsError) as ERROR:
        validate.array(parameter, NAME, lower_bound=lower_bound,
            upper_bound=upper_bound)

    assert str(ERROR.value) == (f"The upper_bound argument "
        f"({upper_bound:.4f}) must be greater than the lower_bound "
        f"argument ({lower_bound:.4f}).")


def test_array_bounds_value_low_exclusive_false() -> None:
    parameter = np.random.normal(0, 1, 3)
    lower_bound = 200
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, lower_bound=lower_bound)
    
    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between or equal to {lower_bound:.4f} and inf.")


def test_array_bounds_single_value_low_exclusive_false() -> None:
    parameter = np.array([-1, 0, 1])
    lower_bound = -0.5
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, lower_bound=lower_bound, 
            exclusive=False)

    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between or equal to {lower_bound:.4f} and inf.")


def test_array_bounds_value_low_exclusive_true() -> None:
    parameter = np.random.normal(0, 1, 3)
    lower_bound = np.amin(parameter)
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, lower_bound=lower_bound, 
            exclusive=True)
    
    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between {lower_bound:.4f} and inf.")


def test_array_bounds_single_value_low_exclusive_true() -> None:
    parameter = np.array([-1, 0, 1])
    lower_bound = -1
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, lower_bound=lower_bound, 
            exclusive=True)

    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between {lower_bound:.4f} and inf.")


def test_array_bounds_value_high_exclusive_false() -> None:
    parameter = np.random.normal(0, 1, 3)
    upper_bound = -200
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME,upper_bound=upper_bound)
    
    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between or equal to -inf and {upper_bound:.4f}.")


def test_array_bounds_single_value_high_exclusive_false() -> None:
    parameter = np.array([-1, 0, 1])
    upper_bound = 0.5
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, upper_bound=upper_bound, 
            exclusive=False)

    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between or equal to -inf and {upper_bound:.4f}.")


def test_array_bounds_value_high_exclusive_true() -> None:
    parameter = np.random.normal(0, 1, 3)
    upper_bound = np.amax(parameter)
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, upper_bound=upper_bound, 
            exclusive=True)
    
    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between -inf and {upper_bound:.4f}.")


def test_array_bounds_single_value_high_exclusive_true() -> None:
    parameter = np.array([-1, 0, 1])
    upper_bound = 1
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        validate.array(parameter, NAME, upper_bound=upper_bound, 
            exclusive=True)

    assert str(ERROR.value) == (f"The {NAME} argument must contain solely "
        f"values between -inf and {upper_bound:.4f}.")


def test_array_bounds_valid_defaults() -> None:
    parameter = np.random.normal(0, 1, 3)
    
    parameter = validate.array(parameter, NAME)
    
    assert True


def test_array_bounds_valid_lower_exclusive_false() -> None:
    parameter = np.random.normal(0, 1, 3)
    lower_bound = -200
    
    parameter = validate.array(parameter, NAME, lower_bound=lower_bound)
    
    assert True


def test_array_bounds_valid_upper_exclusive_false() -> None:
    parameter = np.random.normal(0, 1, 3)
    upper_bound = 200

    parameter = validate.array(parameter, NAME, upper_bound=upper_bound)

    assert True


def test_array_bounds_equal_exclusive_false() -> None:
    parameter = np.zeros(3)
    lower_bound = 0
    upper_bound = 0

    parameter = validate.array(parameter, NAME, lower_bound=lower_bound, 
        upper_bound=upper_bound, exclusive=False)

    assert True

""" ARRAY DIMENSIONS """
""" ---------------- """


def test_array_dimensions_invalid_number() -> None:
    parameter = np.ones(10)
    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME, num_dimensions="invalid")

    assert str(ERROR.value) == "num_dimensions is not of type integer"


def test_array_dimensions_valid_params_1D() -> None:
    parameter = np.ones(10)
    num_dimensions = 1

    parameter = validate.array(parameter, NAME, num_dimensions=num_dimensions)

    assert len(parameter.shape) == num_dimensions


def test_array_dimensions_valid_params_2D() -> None:
    parameter = np.ones((1,1))
    num_dimensions = 2

    parameter = validate.array(parameter, NAME, num_dimensions=num_dimensions)

    assert len(parameter.shape) == num_dimensions


def test_array_dimensions_invalid_params() -> None:
    parameter = np.ones((1,1))
    test_dimensions = len(parameter.shape)
    num_dimensions = 1

    with pytest.raises(InvalidDimensionsError) as ERROR:
        validate.array(parameter, NAME, num_dimensions=num_dimensions)
        
    assert str(ERROR.value) == (f"The {NAME} ({test_dimensions} dimensions) "
        f"should have {num_dimensions} dimensions.")

""" ARRAY DTYPE """
""" ----------- """


def test_array_type_valid() -> None:
    parameter = np.ones(12)
    dtype = str(parameter.dtype)

    validate.array(parameter, NAME, dtype=dtype)

    assert True


def test_array_type_valid_string_type() -> None:
    parameter = np.ones(12).astype(str)
    dtype = str(parameter.dtype)

    parameter = validate.array(parameter, NAME, dtype=dtype)

    assert True


def test_array_type_invalid() -> None:
    parameter = np.ones(12)
    test_type = str(parameter.dtype)
    dtype = str(parameter.astype(str).dtype)

    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME, dtype=dtype)

    assert str(ERROR.value) == f"{NAME} ({test_type}) is not of type {dtype}"


def test_array_type_invalid_input() -> None:
    parameter = np.ones(12)
    invalid_dtype_parameter = True

    with pytest.raises(TypeError) as ERROR:
        validate.array(parameter, NAME, dtype=invalid_dtype_parameter)

    assert str(ERROR.value) == "dtype is not of type string"



###############################################################################
############################# same_shape_arrays() #############################
###############################################################################


def test_same_shape_arrays() -> None:
    arrays_list = [np.zeros(1), np.ones(1)]
    names_list = [NAME, NAME]
    validate.same_shape_arrays(arrays_list, names_list)
    assert True


def test_same_shape_arrays_arrays_list_size_1() -> None:
    arrays_list = [np.zeros(1)]
    names_list = [NAME]
    
    with pytest.raises(AttributeError) as ERROR:
        validate.same_shape_arrays(arrays_list, names_list)

    message = "arrays_list argument should contain at least two arrays"
    assert str(ERROR.value) == message


def test_same_shape_arrays_names_list_wrong_length() -> None:
    arrays_list = [np.zeros(1), np.ones(1)]
    names_list = [NAME]
    
    with pytest.raises(AttributeError) as ERROR:
        validate.same_shape_arrays(arrays_list, names_list)
    
    assert str(ERROR.value) == (f"arrays_list ({len(arrays_list)}) should "
        f"have the same length as names_list ({len(names_list)})")


def test_same_shape_arrays_wrong_shapes() -> None:
    arrays_list = [np.zeros(2), np.ones(1)]
    names_list = [NAME, NAME+NAME]

    with pytest.raises(InvalidShapeError) as ERROR:
        validate.same_shape_arrays(arrays_list, names_list)
    
    message = ""
    for name, array in zip(names_list, arrays_list):
        message = message + f"{name} {str(array.shape)}, "
    message = message[:-2] + " should all have the same shape."
    
    assert str(ERROR.value) == message


def test_same_shape_arrays_not_array() -> None:
    arrays_list = [np.zeros(2), "string"]
    names_list = [NAME, NAME+NAME]

    with pytest.raises(TypeError) as ERROR:
        validate.same_shape_arrays(arrays_list, names_list)

    assert str(ERROR.value) == f"{NAME+NAME} is not of type np.ndarray"



###############################################################################
################################ class_object() ###############################
###############################################################################


def test_class_int_object() -> None:
    instance = 4
    object_class = int
    class_name = "int"
    validate.class_object(instance, NAME, object_class, class_name)
    assert True


def test_class_float_object() -> None:
    instance = 4.
    object_class = float
    class_name = "float"
    validate.class_object(instance, NAME, object_class, class_name)
    assert True


def test_class_array_object() -> None:
    instance = np.array([4.])
    object_class = np.ndarray
    class_name = "numpy array"
    validate.class_object(instance, NAME, object_class, class_name)
    assert True


def test_class_unequal_class() -> None:
    instance = 4
    object_class = str
    class_name = "string"
    
    with pytest.raises(TypeError) as ERROR:
        validate.class_object(instance, NAME, object_class, class_name)

    assert str(ERROR.value) == f"{NAME} is not of type {class_name}"