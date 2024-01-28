import time

from src.read_compact import read_binary_file
from src.filters import filter_total_energy
import src.filters as filters
from src.detector_features import calculate_total_energy


if __name__ == "__main__":
    # Path to the binary file
    binary_file_path = "P:/Valencia/I3M/Proyectos/ForTheGroup/event_petsys/data/ERC/david_data_coinc.ldat"

    # test the time taken to read the entire file using read_binary_file function
    start_time = time.time()
    event_count = 0
    for event in read_binary_file(binary_file_path):
        det1, det2 = event
        det1_en = calculate_total_energy(det1)
        en_filter = filter_total_energy(det1_en)
        # print(f"En filter: {filter_total_energy(det1, 50)}")
        # print(f"Lenghts: {len(det1)}, {len(det2)}")
        # time.sleep(1)
        event_count += 1
        if event_count % 10000 == 0:
            print(f"Events processed: {event_count}")
        pass
    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {filters.EVT_COUNT_T}")
    print(f"Events passing the filter: {filters.EVT_COUNT_F}")
