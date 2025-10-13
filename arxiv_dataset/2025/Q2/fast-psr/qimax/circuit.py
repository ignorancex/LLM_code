from . import converter

class U():
    def __init__(self, params_form, tensor_form, index = 0):
        self.params_form = params_form
        self.tensor_form = tensor_form
        self.matrix_form = None
        self.index = index
    def plus_params_form(self, plus: int):
        # add plus to the element of params_form which different from -999
        params_form = [-999, -999, 1.0, -999]
        self.params_form = [param if param == -999 else param+plus for param in params_form]
        return self
    def compare(self, us):
        for u in us:
            if self.params_form == u.params_form and self.tensor_form == u.tensor_form:
                self.index = u.index
                self.matrix_form = u.matrix_form
                return True
        return False
    def len_params(self):
        return len([param for param in self.params_form if param != -999])
    def to_matrix(self):
        if self.matrix_form is None:
            self.matrix_form = converter.string_to_matrix(self.params_form, self.tensor_form)
        return self.matrix_form