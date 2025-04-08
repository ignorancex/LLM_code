from sqlitedict import SqliteDict
from ManageCONST import readCONST

class DataManager:
    def __init__(self):
        CONST = readCONST()
        self.file_name = f"Database/Data{CONST['SubjectID']}"

    def save(self, key, value):
        """Save a key-value pair to the database."""
        try:
            with SqliteDict(self.file_name) as mydict:
                mydict[key] = value  # Store the value
                mydict.commit()      # Commit to flush the data
        except Exception as ex:
            print("Error during storing data (Possibly unsupported):", ex)

    def load(self, key):
        """Load a value from the database using the specified key."""
        try:
            with SqliteDict(self.file_name) as mydict:
                return mydict[key]  # Directly return the value
        except KeyError:
            print(f"Key '{key}' not found.")
        except Exception as ex:
            print("Error during loading data:", ex)

    def loadAll(self):
        """Load and print all key-value pairs in the database."""
        try:
            with SqliteDict(self.file_name) as mydict:
                for key, item in mydict.items():
                    print(f"{key}: {item}")
                return mydict
        except Exception as ex:
            print("Error during loading data:", ex)

    def loadData(self, length):
        """Load a specific number of records from the database."""
        data = []
        try:
            with SqliteDict(self.file_name) as mydict:
                max_key = max(int(k) for k in mydict.keys()) + 1
                start_key = max(0, max_key - length)
                print(f"Loading data from keys {start_key} to {max_key}")

                for key in range(start_key, max_key):
                    data.append(mydict.get(str(key)))  # Append each value

                return data
        except Exception as ex:
            print("Error during loading data:", ex)

    def clear(self):
        """Clear all data from the database."""
        try:
            with SqliteDict(self.file_name) as mydict:
                mydict.clear()
        except Exception as ex:
            print("Error during clearing data:", ex)

# Example usage
if __name__ == "__main__":
    data_manager = DataManager()
    # Example of saving and loading data
    # data_manager.save(3, [76, 888, 998, 87, 334])
    # result = data_manager.load(10)
    # print(result)
    # data_manager.loadAll()
    # data_manager.clear()
    data = data_manager.loadData(length=10)
    print(len(data))
    
    # Example of plotting data
    # from matplotlib import pyplot as plt
    # plt.plot(data)
    # plt.savefig("./PPG")
