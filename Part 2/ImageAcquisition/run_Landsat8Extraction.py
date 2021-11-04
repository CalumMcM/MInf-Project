import logging
import threading
import time
import ee
from src.ExtractLandsat8 import python2PythonExtraction

def thread_function(start_date, num_imgs, seed, quad_num):

    logging.info("Thread %s: Start_Date: %s num_imgs: %s seed: %s quad_num: %s starting", seed, start_date, num_imgs, seed, quad_num)

    python2PythonExtraction(num_imgs, seed, quad_num, start_date)

def main():
    
    start_dates = ['2015-05-01', '2016-05-01', '2017-05-01', '2018-05-01', '2019-05-01']
    end_dates = ['2015-10-15', '2016-10-15', '2017-10-15', '2018-10-15', '2019-10-15']

    CUR_START_DATE = start_dates[0] # THIS SHOULD BE THE SAME AS IT IS IN ExtractLandsat8.py
    NUM_IMGS = 5
    NUM_THREADS = 5
    #python2PythonExtraction(10, 0, 1, '2015-05-01', '2015-10-15')

    with open('progress.txt', 'r') as f:
        START_IMAGE = f.read()
        START_IMAGE = int(START_IMAGE)
        f.close()

        print ("Starting Image: {}".format(START_IMAGE))

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    threads = []
    for QUAD_NUM in range(1, 5):
        
        # Starts y/NUM_IMGS threads where range(x,y,z)
        for SEED in range(START_IMAGE, NUM_IMGS*NUM_THREADS, NUM_IMGS):
            x = threading.Thread(target=thread_function, args=(CUR_START_DATE, NUM_IMGS, SEED, QUAD_NUM))
            x.start()
            threads.append(x)

        for SEED in range(0, len(threads)):
            threads[SEED].join()

            # Update logging file of downloaded images
            with open('progress.txt', 'w') as f:
                f.write(str(((SEED+1)*NUM_IMGS)))
            f.close()
            # Save the number of images that were just successfuly downloaded 
            logging.info("Thread %s finished : Images %s-%s saved", SEED, SEED*NUM_IMGS, ((SEED+1)*NUM_IMGS)-1)
       
        # Clear the thread array
        threads = []

    logging.info("Main    : all done")
    

if __name__ == "__main__":
    ee.Authenticate()

    ee.Initialize()

    main()
    