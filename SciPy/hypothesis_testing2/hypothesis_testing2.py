import numpy as np
import "fetchmaker"
from scipy.stats import binom_test
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

rottweiler_tl = fetchmaker.get_tail_length("rottweiler")

print(np.mean(rottweiler_tl))
print(np.std(rottweiler_tl))

"""
Over the years, we have seen that we expect 8% of dogs in the FetchMaker system to be rescues. We want to know if whippets are significantly more or less likely to be a rescue.
"""
whippet_rescue = fetchmaker.get_is_rescue("whippet")
num_whippet_rescues = np.count_nonzero(whippet_rescue)
print(num_whippet_rescues)
num_whippets = np.size(whippet_rescue)
print(num_whippets)

pval = binom_test(num_whippet_rescues, n=num_whippets, p=0.08)
print(pval)

"""
Three of our most popular mid-sized dog breeds are whippets, terriers, and pitbulls. Is there a significant difference in the average weights of these three dog breeds? Perform a comparative numerical test to determine if there is a significant difference.
"""
whippet_weight = fetchmaker.get_weight("whippet")
terrier_weight = fetchmaker.get_weight("terrier")
pitbull_weight = fetchmaker.get_weight("pitbull")
print(np.size(whippet_weight))
print(np.size(terrier_weight))
print(np.size(pitbull_weight))
tstat, pval = f_oneway(whippet_weight, terrier_weight, pitbull_weight)
print(pval)

v = np.concatenate([whippet_weight, terrier_weight, pitbull_weight])
labels = ['whippet_weight'] * len(whippet_weight) + ['terrier_weight'] * len(terrier_weight) + ['pitbull_weight'] * len(pitbull_weight)

tukey_results = pairwise_tukeyhsd(v, labels, 0.05)
print(tukey_results)

"""
We want to see if "poodle"s and "shihtzu"s have significantly different color breakdowns.
"""
poodle_colors = fetchmaker.get_color("poodle")
shihtzu_colors = fetchmaker.get_color("shihtzu")
color_table = [
  [np.count_nonzero(poodle_colors == "black"), np.count_nonzero(shihtzu_colors == "black")],
  [np.count_nonzero(poodle_colors == "brown"), np.count_nonzero(shihtzu_colors == "brown")],
  [np.count_nonzero(poodle_colors == "gold"), np.count_nonzero(shihtzu_colors == "gold")],
  [np.count_nonzero(poodle_colors == "grey"), np.count_nonzero(shihtzu_colors == "grey")],
  [np.count_nonzero(poodle_colors == "white"), np.count_nonzero(shihtzu_colors == "white")]
]
print(color_table)
chi2, pval, dof, expected = chi2_contingency(color_table)
print(pval)
