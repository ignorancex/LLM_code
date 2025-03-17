import os
import time
import json
import operator

from flask import jsonify, request
from flask_restful import Resource, reqparse

from xoa.commons.logger import * 


class ObservedErrors(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']

        super(ObservedErrors, self).__init__()

    def get(self, space_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401
        
        space = self.sm.get_space(space_id)
        if space == None:
            return "Search space {} is not available".format(space_id), 404

        errors = []
        for c_id in space.get_completions():
            c_id = int(c_id)
            err = {"id" : c_id}
            err['num_epochs'] = space.get_train_epoch(c_id) 
            err["valid_error"] = space.get_errors(c_id, error_type='valid')
            err["test_error"] = space.get_errors(c_id, error_type='test')
            err["order"] = space.get_search_order(c_id)
                       
            errors.append(err)
        errors.sort(key=operator.itemgetter('test_error'))        
        
        return errors, 200
