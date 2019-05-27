def simple_accuracy(preds, labels):
	cnt = 0
	for i in range(len(preds)):
		if preds[i] == labels[i]:
			cnt += 1
	return cnt / len(preds)

def full_accuracy(preds, labels):
	# [0 matches, exactly 1 match, exactly 2 matches, exactly 3 matches, exactly 4 matches]
	num_same_array = [0,0,0,0,0]
	for i in range(len(preds)):
		curr_cnt = 0
		if (preds[i][0] == labels[i][0]): curr_cnt += 1
		if (preds[i][1] == labels[i][1]): curr_cnt += 1
		if (preds[i][2] == labels[i][2]): curr_cnt += 1
		if (preds[i][3] == labels[i][3]): curr_cnt += 1
		num_same_array[curr_cnt] += 1

	# cumulative counts 
	# [0 matches, at least 1 match, at least 2 matches, at least 3 matches, at least 4 matches]
	num_same_array[3] += num_same_array[4]
	num_same_array[2] += num_same_array[3]
	num_same_array[1] += num_same_array[2]

	for i in range(len(num_same_array)):
		num_same_array[i] /= len(preds)
	return num_same_array

def mbti_accuracy(preds, labels):
	mbti_accuracy_array = [0, 0, 0, 0]
	for i in range(len(preds)):
		if (preds[i][0] == labels[i][0]): mbti_accuracy_array[0] += 1
		if (preds[i][1] == labels[i][1]): mbti_accuracy_array[1] += 1
		if (preds[i][2] == labels[i][2]): mbti_accuracy_array[2] += 1
		if (preds[i][3] == labels[i][3]): mbti_accuracy_array[3] += 1
	for i in range(len(mbti_accuracy_array)):
		mbti_accuracy_array[i] /= len(preds)
	return mbti_accuracy_array


def convert_to_types(preds, labels, label_list):
	label_map = {}
	for i in range(len(label_list)):
		label_map[i] = label_list[i]

	new_preds = []
	for i in preds:
		new_preds.append(label_map[i])

	new_labels = []
	for i in labels:
		new_labels.append(label_map[i])

	return new_preds, new_labels


def compute_metrics(preds, labels, label_list):

	preds, labels = convert_to_types(preds, labels, label_list)
	simple_acc = simple_accuracy(preds, labels)
	metrics = {"simple_acc": simple_acc}

	if (len(preds[0]) == 1):
		return metrics
	else:
		full_acc = full_accuracy(preds, labels)
		mbti_acc = mbti_accuracy(preds, labels)
		metrics["at_least_1_match"] = full_acc[1] 
		metrics["at_least_2_matches"] = full_acc[2] 
		metrics["at_least_3_matches"] = full_acc[3]
		metrics["all_4_matches"] = full_acc[4]
		metrics["E/I"] = mbti_acc[0]
		metrics["N/S"] = mbti_acc[1]
		metrics["F/T"] = mbti_acc[2]
		metrics["P/J"] = mbti_acc[3]
		return metrics