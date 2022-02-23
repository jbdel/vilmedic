from .inference_from_script import get_rewards
import json

report = open("/home/jb/Documents/vilmedic/vilmedic/vilmedic/blocks/scorers/radgraph/reports/all/4126199_6017638_post.txt").read()
print(report)
hypothesis_report_list = [report]

reward_list, hyp_annot, ref_annot = get_rewards(hypothesis_report_list, hypothesis_report_list, 0.5,
                          0.5, "partial", -1)


print(json.dumps(hyp_annot, indent=4))
