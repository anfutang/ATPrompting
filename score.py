import os
import time
import datetime
import logging
import pickle
from opt import get_args
from scorer.scorer import Scorer
from utils.utility import build_dst_folder

if __name__ == "__main__":
    args = get_args()

    logging_dir, output_dir = build_dst_folder(args)

    logging_filename = os.path.join(logging_dir,f"{args.score_type}_score.log")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=logging_filename,filemode='w')

    logging.info("*Start time:"+str(datetime.datetime.now()))

    start_time = time.time()

    scores = Scorer(args).score()
    
    with open(os.path.join(output_dir,f"{args.score_type}_result.pkl"),"wb") as f:
        pickle.dump(scores,f,pickle.HIGHEST_PROTOCOL)

    logging.info(f"Scores saved under {output_dir}.")

    end_time = time.time()
    logging.info(f"Finished: {end_time-start_time} s.")
