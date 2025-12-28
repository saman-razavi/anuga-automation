#!/usr/bin/env python

from anuga.utilities.sww_merge import sww_merge_parallel

# Base name WITHOUT _P4 and WITHOUT .sww
DOMAIN_NAME = "20140702000000_Shellmouth_flood_12_days"
NP = 20  # number of processors

if __name__ == "__main__":
    print("Merging SWW files for domain:", DOMAIN_NAME)
    print("Expecting files like:", f"{DOMAIN_NAME}_P{NP}_0.sww ...")
    sww_merge_parallel(DOMAIN_NAME, NP, verbose=True, delete_old=False)
    print("Done. You should now have:", DOMAIN_NAME + ".sww")

