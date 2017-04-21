# -*- coding: utf-8 -*-
"""
@author: Evgeny BAZAROV <baz.evgenii@gmail.com>
@brief:

"""

import os
import logging
import requests

DD_TIMEOUT = 86400  # 24h (86400 sec) call maximum :)

API_METHODS_URL = {
    "0.1": {
        "info": "/info",
        "services": "/services",
        "train": "/train",
        "predict": "/predict"
    }
}


class DD(object):
    """HTTP requests to the DeepDetect server

    """

    # return types
    RETURN_JSON = 0
    RETURN_TEXT = 1
    RETURN_NONE = 2

    __HTTP = 0
    __HTTPS = 1

    def __init__(self, host="localhost", port=8080, proto=0, apiversion="0.1"):
        """ DD class constructor
        Parameters:
        host -- the DeepDetect server host
        port -- the DeepDetect server port
        proto -- user http (0,default) or https connection
        """
        self.apiversion = apiversion
        self.__urls = API_METHODS_URL[apiversion]
        self.__host = host
        self.__port = port
        self.__proto = proto
        self.__returntype = self.RETURN_JSON
        if proto == self.__HTTP:
            self.__ddurl = 'http://%s:%d' % (host, port)
        else:
            self.__ddurl = 'https://%s:%d' % (host, port)

    def set_return_format(self, f):
        assert f == self.RETURN_JSON or f == self.RETURN_TEXT or f == self.RETURN_NONE
        self.__returntype = f

    def __return_data(self, r):
        if self.__returntype == self.RETURN_JSON:
            return r.json()
        elif self.__returntype == self.RETURN_TEXT:
            return r.text
        else:
            return None

    def get(self, method, params=None):
        """GET to DeepDetect server """
        url = self.__ddurl + method
        r = requests.get(url=url, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    def put(self, method, params):
        """PUT request to DeepDetect server"""
        url = self.__ddurl + method
        r = requests.put(url=url, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    def post(self, method, params):
        """POST request to DeepDetect server"""
        url = self.__ddurl + method
        r = requests.post(url=url, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    def delete(self, method, params):
        """DELETE request to DeepDetect server"""
        url = self.__ddurl + method
        r = requests.delete(url=url, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    # API methods
    def info(self):
        """Info on the DeepDetect server"""
        return self.get(self.__urls["info"])

    # API services
    def put_service(self, sname, model, description, mllib, parameters_input, parameters_mllib, parameters_output,
                    mltype='supervised'):
        """
        Create a service
        Parameters:
        sname -- service name as a resource
        model -- dict with model location and optional templates
        description -- string describing the service
        mllib -- ML library name, e.g. caffe
        parameters_input -- dict of input parameters
        parameters_mllib -- dict ML library parameters
        parameters_output -- dict of output parameters
        """
        params = {"description": description,
                  "mllib": mllib,
                  "type": mltype,
                  "parameters": {"input": parameters_input,
                                 "mllib": parameters_mllib,
                                 "output": parameters_output},
                  "model": model}
        return self.put(self.__urls["services"] + '/%s' % sname, params)

    def get_service(self, sname):
        """
        Get information about a service
        Parameters:
        sname -- service name as a resource
        """
        return self.get(self.__urls["services"] + '/%s' % sname)

    def delete_service(self, sname, clear=None):
        """
        Delete a service
        Parameters:
        sname -- service name as a resource
        clear -- 'full','lib' or 'mem', optionally clears model repository data
        """
        params = None
        if clear:
            params = {"clear": clear}
        return self.delete(self.__urls["services"] + '/%s' % sname, params)

    # API train
    def post_train(self, sname, data, parameters_input, parameters_mllib, parameters_output, async=True):
        """
        Creates a training job
        Parameters:
        sname -- service name as a resource
        async -- whether to run the job as non-blocking
        data -- array of input data / dataset for training
        parameters_input -- dict of input parameters
        parameters_mllib -- dict ML library parameters
        parameters_output -- dict of output parameters
        """
        params = {"service": sname,
                  "async": async,
                  "parameters": {"input": parameters_input,
                                 "mllib": parameters_mllib,
                                 "output": parameters_output},
                  "data": data}
        return self.post(self.__urls["train"], params)

    def get_train(self, sname, job=1, timeout=0, measure_hist=False):
        """
        Get information on a non-blocking training job
        Parameters:
        sname -- service name as a resource
        job -- job number on the service
        timeout -- timeout before obtaining the job status
        measure_hist -- whether to return the full measure history (e.g. for plotting)
        """
        params = {"service": sname,
                  "job": str(job),
                  "timeout": str(timeout)}
        if measure_hist:
            params["parameters.output.measure_hist"] = measure_hist
        return self.get(self.__urls["train"], params)

    def delete_train(self, sname, job=1):
        """
        Kills a non-blocking training job
        Parameters:
        sname -- service name as a resource
        job -- job number on the service
        """
        params = {"service": sname,
                  "job": str(job)}
        return self.delete(self.__urls["train"], params)

    # API predict
    def post_predict(self, sname, data, parameters_input, parameters_mllib, parameters_output):
        """
        Makes prediction from data and model
        Parameters:
        sname -- service name as a resource
        data -- array of data URI to predict from
        parameters_input -- dict of input parameters
        parameters_mllib -- dict ML library parameters
        parameters_output -- dict of output parameters
        """
        params = {"service": sname,
                  "parameters": {"input": parameters_input,
                                 "mllib": parameters_mllib,
                                 "output": parameters_output},
                  "data": data}
        return self.post(self.__urls["predict"], params)

# test
if __name__ == '__main__':
    dd = DD()
    dd.set_return_format(dd.RETURN_JSON)
    inf = dd.info()
    print(inf)
