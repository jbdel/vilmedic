from radgraph.inference_from_script import get_rewards

report = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
hypothesis_report_list = [report, "", "a", report]

report_2 = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The heart is clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
reference_report_list = [report_2, report_2, report_2, report_2]

reward_list = get_rewards(hypothesis_report_list, reference_report_list, 0.5,
                          0.5, "partial", -1)


print(reward_list)  # [0.8666666666666667, 0, 0, 0.8666666666666667]
