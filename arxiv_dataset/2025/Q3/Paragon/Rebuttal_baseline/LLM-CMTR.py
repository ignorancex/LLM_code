rec_prompt = "You are a recommendation expert who receives a user's chronological purchase history and provides the next recommended item from a given set of candidate products. "\
"Please note that you need to consider the \"accuracy\" and \"diversity\" of recommendations comprehensively.. "\
"Their definitions are as follows: " \
"Objective 1: Accuracy: Ensure that the recommended items are highly relevant to the user's interests and needs, thereby ensuring the accuracy of the recommendations. To measure the effectiveness of your recommendations, it will use nDCG as the evaluation metric. "\
"Objective 2: Diversity: Ensure that the recommended content is diverse, avoiding excessive recommendations of similar items. To achieve this, it will use Alpha-nDCG as the evaluation metric, penalizing overly similar recommended items and encouraging a diverse range of content in the recommendation list. "\
"Now, I will provide the current user's purchase history and the set of candidate products. Purchase history: [-history-]. Candidate product set: [-candidates-]." \
"Please rank these candidates and give [-out_num-] item as recommendationns and make them both diverse and accurately relevant with the history preference. "\
"To achieve this, consideri the priority of these two objectives according to the given priority weights (\"accuracy\":[-w_accuracy-] and \"diversity\":[-w_diversity-]) "\
"Split your output with line break. You MUST rank and output 10 items as recommendations. " \
"You can not generate candidates that are not in the given candidate set."
