import os
import time
import json

from flask import jsonify, request
from flask_restful import Resource, reqparse

from xoa.commons.logger import * 


class ObservedError(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']

        super(ObservedError, self).__init__()

    def get(self, space_id, sample_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401
        
        space = self.sm.get_space(space_id)
        if space == None:
            return "Search space {} is not available".format(space_id), 404

        sample_id = int(sample_id)
        error = {"id": sample_id}
        error['num_epochs'] = space.get_train_epoch(sample_id)
        error["valid_error"] = space.get_errors(sample_id, error_type='valid')
        error["test_error"] = space.get_errors(sample_id, error_type='test')
        
        
        return error, 200 
    
    def put(self, space_id, sample_id):
        parser = reqparse.RequestParser()        
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("value", location='args', type=float)
        parser.add_argument("error_type", location='args', type=str)
        parser.add_argument("num_epochs", location='args', type=int, default=1)
        args = parser.parse_args()

        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        space = self.sm.get_space(space_id)
        if space is None:
            return "Space {} not found".format(space_id), 404
        else:
            try:
                if space_id != "active":
                    self.sm.set_space_status(space_id, "active")
                sample_id = int(sample_id)
                space.update_error(sample_id, args["value"], args["num_epochs"], args["error_type"])
                error = {"id": sample_id }
                key = '{}_error'.format(args["error_type"])
                error[key] = space.get_errors(sample_id, error_type=args["error_type"])
                error["num_epochs"] = args["num_epochs"]
                
                return error, 202

            except Exception as ex:
                warn("Error update exception: {}".format(ex))
                return "Invalid request:{}".format(ex), 400         

