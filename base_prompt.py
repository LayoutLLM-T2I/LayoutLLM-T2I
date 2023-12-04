

def add_prefix(example, query):
    if example != '': # few_shot
        prompt =  ('Now you are an assistant to help me design a layout given a description. '
                'Concretely, a layout denotes a set of "object: bounding box" item. '
                '"object" means any object name in the world, while "bounding box" is formulated as [x, y, w, h], where "x, y" denotes the top left coordinate of the bounding box, "w" denotes the width, and "h" denotes the height. '
                'The six values "x, y, w, h, x+w, y+h" are all larger than 0 and smaller than 1. '
                'Next, I will give you several examples for you to understand this task.'
                f'\n{example}\n{query}')
    else: # zero-shot
        prompt =  ('Now you are an assistant to help me design a layout given a description. '
                'Concretely, a layout denotes a set of "object: bounding box" item. '
                '"object" means any object name in the world, while "bounding box" is formulated as [x, y, w, h], where "x, y" denotes the top left coordinate of the bounding box, "w" denotes the width, and "h" denotes the height. '
                'The six values "x, y, w, h, x+w, y+h" are all larger than 0 and smaller than 1. '
                'Next, I will give you an input which describes an image, and then you should give me an output with the format "'
                '\noutput:\nobject: [x, y, w, h], \nobject: [x, y, w, h],\n...\n"'
                f'\n{example}\n{query}')
    return prompt


def build_prompt(shot_cand, test_example, args):
    cap = test_example['captions']
    in_context_str = ''
    for i, cur_cand in enumerate(shot_cand):
        cap_train = cur_cand['captions']
        input_str = '\ninput: ' + cap_train + '\n'

        labels = cur_cand['label']
        boxes = cur_cand['bbox']
        l_box_str_all = ['output: ']
        for jj in range(len(labels)):
            xc, yc, w, h = boxes[jj]
            box = [xc - w/2, yc - h/2, w, h]
            box = [round(x, 2) for x in box]
            l_box_str = labels[jj] + ': ' + str(box)
            l_box_str_all.append(l_box_str)
        l_box_str_all = '\n'.join(l_box_str_all)
        io_str = input_str + l_box_str_all + '\n'
        in_context_str += io_str

    query_str = f'input: {cap} (No explanation. Must give an output or try to imagine a possible output even if the given description is incomplete. )'
    prompt_input = add_prefix(in_context_str, query_str)
    
    return prompt_input