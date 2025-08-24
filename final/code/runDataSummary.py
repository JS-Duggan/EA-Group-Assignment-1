import pandas
import statistics

def calculate_statistics(data):
    """
    Calcualtes 5 key statistical values for a set of data and displays them on the terminal

    Args:
        data (list[int]): Data through which the calculations are based on
    """
    
    print(' - mean = ' + str(statistics.mean(data)))
    print(' - median = ' + str(statistics.median(data)))
    print(' - mode = ' + str(statistics.mode(data)))

    print(' - max = ' + str(max(data)))
    print(' - min = ' + str(min(data)))

def process_data(path):
    """
    Processes the data at the file path and caculates key statistical values for the three different methods
    Results are published on the terminal

    Args:
        path (string): File path to the raw data to be processed
    """
    
    file = pandas.read_csv(path)
    jump_costs = []
    exchange_costs = []
    inverse_costs = []

    # Process the relevant data from each row
    for index, row in file.iterrows():
        jump_costs.append(int(row['Jump - cost']))
        exchange_costs.append(int(row['Exchange - cost']))
        inverse_costs.append(int(row['Inverse - cost']))

    # Calcualte the data and display it to terminal
    print(path + ':')
    print('Jump: ')
    calculate_statistics(jump_costs)
    print('Exchange: ')
    calculate_statistics(exchange_costs)
    print('Inverse: ')
    calculate_statistics(inverse_costs)
