# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:23:17 2020

@author: nkraj
"""

import glassdoor_scraper as gs
import pandas as pd

path = "C:/Users/nkraj/Documents/Data Science/ds_salary_proj/chromedriver"

df = gs.get_jobs('data scientist', 1000, False, path, 5)

df.to_csv('glassdoor_jobs.csv', index=False)
