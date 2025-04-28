#!/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Module to access the EAGLE public database
# John Helly 2015 for the Virgo Consortium

from getpass import getpass
from http import cookiejar
import numpy as np
import re
from urllib import parse, request

# Mapping between SQL and numpy types
numpy_dtype = {
    "real"     : np.float32,
    "float"    : np.float64,
    "int"      : np.int32,
    "bigint"   : np.int64,
    "char"     : np.dtype("|S256"),
    "nvarchar" : np.dtype("|S256")
    }

# Cookie storage - want to avoid creating a new session for every query
cookie_file = "sql_cookies.txt"
cookie_jar = cookiejar.LWPCookieJar(cookie_file)
try:
    cookie_jar.load(ignore_discard=True)
except IOError:
    pass


class WebDBConnection:
    def __init__(self, username, password=None):
        """Class to store info required to connect to the web server"""
        # Get password if necessary
        if password is None:
            password = getpass()
        # Get URL for the database
        self.db_url = "http://galaxy-catalogue.dur.ac.uk:8080/Eagle"
        # Set up authentication and cookies
        self.password_mgr = request.HTTPPasswordMgrWithDefaultRealm()
        self.password_mgr.add_password(None, self.db_url, username, password)
        self.opener = request.OpenerDirector()
        self.auth_handler   = request.HTTPBasicAuthHandler(self.password_mgr)
        self.cookie_handler = request.HTTPCookieProcessor(cookie_jar)

    def execute_query(self, sql):
        """Run an SQL query and return the result as a record array"""
        url = self.db_url + "?" \
            + parse.urlencode({'action': 'doQuery', 'SQL': sql})
        request.install_opener(
            request.build_opener(self.auth_handler, self.cookie_handler))
        response = request.urlopen(url)
        cookie_jar.save(ignore_discard=True)

        # Check for OK response
        line = response.readline()
        if line != "#OK\n":
            raise Exception(response.readlines())

        # Skip rows until we reach QUERYTIMEOUT
        while True:
            line = response.readline()
            if line == "":
                raise Exception(
                    "Unexpected end of file while reading result header")
            elif line.startswith("#QUERYTIMEOUT"):
                break

        # Skip QUERYTIME
        if not(response.readline().startswith("#QUERYTIME")):
            raise Exception("Don't understand result header!")

        # Read column info
        # (also discards line with full list of column names)
        columns = []
        while True:
            line = response.readline()
            if line[0] != "#":
                column_names = line
                break
            else:
                m = re.match(
                    "^#COLUMN ([0-9]+) name=([\w]+) JDBC_TYPE=(-?[0-9]+)" \
                    " JDBC_TYPENAME=([\w]+)$", line)
                if m is not None:
                    columns.append(m.groups())
                else:
                    raise Exception("Don't understand column info: "+line)

        # Construct record type for the output
        dtype = np.dtype([(col[1],numpy_dtype[col[3]]) for col in columns])

        # Return the data as a record array
        return np.genfromtxt(response, dtype=dtype, delimiter=",")

    def fetch_docs(self, table):
        """Return a list of strings containing the documentation page
           for the specified table"""
        url = self.db_url + "/Help?" \
            + parse.urlencode({'page': "databases/"+"Eagle"+"/"+table})
        request.install_opener(
            request.build_opener(self.auth_handler, self.cookie_handler))
        response = request.urlopen(url)
        cookie_jar.save(ignore_discard=True)
        return response.readlines()


def connect(user, password=None):
    """Connect to EAGLE database and return a connection object"""
    return WebDBConnection(user, password)


def execute_query(con, sql):
    return con.execute_query(sql)

