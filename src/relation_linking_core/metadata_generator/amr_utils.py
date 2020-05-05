import penman


class AMRUtils:

    @classmethod
    def fix_amr_graph(cls, amr_string):
        amr_graph = penman.loads(amr_string)
        if len(amr_graph) > 1:
            print('WARNING: enhanced AMR is mis-formatted')
            lines = amr_string.split('\n')
            for i in range(len(lines)):
                if ':entities' in lines[i] and lines[i - 1].strip().endswith(')'):
                    lines[i - 1] = lines[i - 1][:len(lines[i - 1]) - 1]
            new_amr = ' '.join(lines)
            print(new_amr)
            amr_graph = penman.loads(' '.join(lines))[0]
        else:
            amr_graph = amr_graph[0]
        return amr_graph