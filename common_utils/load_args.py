import json


class Params:
    """Class that loads parameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.code_dir)
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            _params = json.load(f)
            self.__dict__.update(_params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            _params = json.load(f)
            self.__dict__.update(_params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


if __name__ == '__main__':
    params = Params('/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json')
    print(params.dict)
