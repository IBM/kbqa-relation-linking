import calendar
import re
from itertools import combinations
from nltk.stem import WordNetLemmatizer


class AMR2Triples:

    lemmatizer = WordNetLemmatizer()
    op_pattern = re.compile("op([0-9]+)")
    ARG_REGEX = re.compile("ARG([0-9]+)")
    propbank_pattern = re.compile("([a-z0-9]+_)*(([a-z]+)-)+(\d\d)")
    non_core_roles = ['accompanier', 'age', 'beneficiary', 'concession', 'condition', 'consist-of', 'destination',
                      'direction', 'domain', 'duration', 'example', 'extent', 'frequency', 'instrument', 'location',
                      'manner', 'medium', 'mod', 'ord', 'part', 'path', 'prep-with', 'purpose', 'quant', 'source',
                      'subevent', 'time', 'topic', 'value']
    conjunctions = ['or', 'and']
    ignored_roles = ['name', 'instance', 'entities', 'entity', 'surface_form', 'type', 'uri']

    @classmethod
    def print_triples(cls, triples, title):
        print('\n{}:\n'.format(title))
        for source, source_id, relation, target, target_id in triples:
            print('{}\t{}\t{}\t{}\t{}'.format(source, source_id, relation, target, target_id))
        print('\n')

    @classmethod
    def concat_name(cls, names_ops):
        name = ''
        for i in range(1, 15):
            if i in names_ops:
                name += ' ' + names_ops[i]
        return name

    @classmethod
    def get_variable_text(cls, var_id, names, dates):
        surface_text = ''
        if var_id in names:
            surface_text = names[var_id]
        elif var_id in dates:
            surface_text = dates[var_id]
        return surface_text.strip()

    @classmethod
    def get_triples(cls, sentence_text, graph, debug=False):

        sentence_text = sentence_text.replace('.', '').replace('?', '')
        token_list = [AMR2Triples.lemmatizer.lemmatize(token) for token in sentence_text.split()]

        triples = graph.triples()

        if debug:
            print("Raw triples:")
            for trip in triples:
                print("{}\t{}\t{}".format(trip[0], trip[1], trip[2]))

        processed_triples = []
        # mappings between variables, their types, and names
        name_var_list, var_to_type, var_to_name, name_to_var = list(), dict(), dict(), dict()
        # amr-unknown variable
        amr_unknown_var = None
        # info for concatenating named entity names.
        grouped_names_ops, names = dict(), dict()
        # map for resolving multi-words written with mod relations
        mod_maps, mod_resolved = dict(), set()
        # handling date time instances
        date_var_list, grouped_var_to_date, dates = list(), dict(), dict()
        # handling ordinals
        ordinal_var_list, ordinal_value = list(), dict()
        # temporal quantity
        temporal_var_list, temporal_value = list(), dict()
        # handle and triples
        and_var_list, and_values = list(), dict()

        # resolve the grouped_names_ops first
        for source, relation, target in triples:
            # instance - type relations
            if relation == 'instance':
                var_to_type[source] = target
                if target == 'amr-unknown':
                    amr_unknown_var = source
                elif target == 'name':
                    name_var_list.append(source)
                elif target == 'date-entity':
                    date_var_list.append(source)
                elif target == 'ordinal-entity':
                    ordinal_var_list.append(source)
                elif target == 'temporal-quantity':
                    temporal_var_list.append(source)
                elif target == 'and':
                    and_var_list.append(source)

            # var - name relations
            elif relation == 'name':
                var_to_name[source] = target
            # collecting all mod relation values
            elif relation == 'mod' and target != 'amr-unknown':
                # we ignore all expressive nodes
                if target == 'expressive':
                    continue
                mod_values = mod_maps.get(source, list())
                mod_values.append(target)
                mod_maps[source] = mod_values
            elif relation in ['year', 'month', 'day', 'weekday'] and source in date_var_list:
                var_to_date_group = grouped_var_to_date.get(source, dict())
                var_to_date_group[relation] = target
                grouped_var_to_date[source] = var_to_date_group
            elif relation == 'value' and source in ordinal_var_list:
                ordinal_value[source] = target
            elif relation == 'quant' and source in temporal_var_list:
                temporal_value[source] = target
            # collecting all op* relations
            elif re.match(AMR2Triples.op_pattern, relation):
                if source in name_var_list:
                    op_pos = int(AMR2Triples.op_pattern.match(relation).group(1))
                    name_ops = grouped_names_ops.get(source, dict())
                    name_ops[op_pos] = str(target).replace("\"", "")
                    grouped_names_ops[source] = name_ops
                elif source in and_var_list:
                    and_ops = and_values.get(source, set())
                    and_ops.add(target)
                    and_values[source] = and_ops

        for var in var_to_name:
            name_to_var[var_to_name[var]] = var

        for var in mod_maps:
            head_word = var_to_type[var]
            if head_word in token_list:
                head_pos = token_list.index(head_word)
                mod_type_var = dict()
                mod_list = list()
                for mod_var in mod_maps[var]:
                    if mod_var in var_to_type:
                        mod_type = var_to_type[mod_var]
                    else:
                        mod_type = mod_var
                    mod_list.append(mod_type)
                    mod_type_var[mod_type] = mod_var
                init_pos = head_pos - (len(mod_list) + 1)
                init_pos = init_pos if init_pos >= 0 else 0
                filtered_tokens = token_list[init_pos:head_pos]
                new_type_tokens = list()
                for token in filtered_tokens:
                    if token in mod_list:
                        mod_resolved.add(mod_type_var[token])
                        new_type_tokens.append(token)
                new_type_tokens.append(head_word)
                new_type = ' '.join(new_type_tokens)
                var_to_type[var] = new_type

        for name_id in grouped_names_ops:
            if name_id in name_to_var:
                names[name_to_var[name_id]] = AMR2Triples.concat_name(grouped_names_ops[name_id])

        for date_id in grouped_var_to_date:
            date_map = grouped_var_to_date[date_id]
            date_list = list()
            if 'day' in date_map:
                date_list.append(str(date_map['day']))
            if 'month' in date_map and date_map['month'] <= 12 and date_map['month'] > 0:
                date_list.append(calendar.month_name[date_map['month']])
            if 'year' in date_map:
                date_list.append(str(date_map['year']))
            dates[date_id] = '/'.join(date_list)

        # process and values
        # TODO this does not fix the issue, this only takes one of the values. This should be fixed in a higher level.
        new_triples = list()
        for source, relation, target in triples:
            if target in and_values:
                all_values = and_values[target]
                for value in all_values:
                    new_triples.append([source, relation, value])
            else:
                new_triples.append([source, relation, target])
        triples = new_triples

        for source, relation, target in triples:
            source_id = source
            target_id = target

            # TODO handle 'interrogative` trees
            if target == 'interrogative':
                continue
            if relation in ['instance', 'name', 'entities', 'entity', 'id', 'type', 'surface_form', 'uri'] or re.match(AMR2Triples.op_pattern, relation) \
                    or source in date_var_list or source in ordinal_var_list or source in temporal_var_list:
                # we have already processed these triples and collected the necessary information
                continue
            if relation == 'mod':
                if target == amr_unknown_var:
                    # sometimes amr-unknown is indirectly attached with a mod relation to a variable
                    amr_unknown_var = source
                    continue
                if target in mod_resolved:
                    continue

            if relation == 'domain':
                if target == amr_unknown_var:
                    # sometimes amr-unknown is indirectly attached with a mod relation to a variable
                    amr_unknown_var = source
                    continue

            if source in var_to_type:
                source = str(var_to_type[source])

            if target in dates:
                target = dates[target]

            if target in var_to_type:
                target = str(var_to_type[target])

            if target in ordinal_value:
                target = str(ordinal_value[target])

            if target in temporal_value:
                target = str(temporal_value[target])

            processed_triples.append([source, source_id, relation, target, target_id])

        if debug:
            AMR2Triples.print_triples(processed_triples, 'Processed triples')
        return processed_triples, var_to_name, var_to_type, names, dates, ordinal_value, temporal_value, amr_unknown_var, graph.top

    @classmethod
    def get_flat_triples(cls, sentence_text, penman_tree):
        triple_info = list()
        frame_args = dict()
        id_to_type = dict()
        reified_to_rel = dict()
        processed_triples, var_to_name, var_to_type, names, dates, ordinal_value, temporal_value, amr_unknown_var, top_node = \
            AMR2Triples.get_triples(sentence_text, penman_tree)
        for subject, source_id, relation, target, target_id in processed_triples:
            id_to_type[source_id] = subject
            id_to_type[target_id] = target
            subject_text, object_text = '', ''
            if source_id in names:
                subject_text = names[source_id]
            elif source_id in dates:
                subject_text = dates[source_id]
                id_to_type[source_id] = 'date-entity'
            if target_id in names:
                object_text = names[target_id]
            elif target_id in dates:
                object_text = dates[target_id]
                id_to_type[target_id] = 'date-entity'
            subject_text = subject_text.strip()
            object_text = object_text.strip()

            # select subjects that are frames
            if re.match(AMR2Triples.propbank_pattern, str(subject)):
                # TODO what should we do when the object is a frame
                if re.match(AMR2Triples.propbank_pattern, str(target)):
                    target = re.match(AMR2Triples.propbank_pattern, str(target)).group(2)
                # we have handled these before (and & or)
                if subject in AMR2Triples.conjunctions or target in AMR2Triples.conjunctions:
                    continue
                args = frame_args.get(source_id, dict())
                if re.match(AMR2Triples.ARG_REGEX, relation) or relation in AMR2Triples.non_core_roles:
                    args[relation] = target_id
                frame_args[source_id] = args
            elif relation not in AMR2Triples.ignored_roles and not re.match(AMR2Triples.ARG_REGEX, relation):
                subject_type = str(var_to_type[source_id]).split()[-1]

                triple = dict()
                triple['subj_text'] = subject_text
                triple['subj_type'] = str(subject).strip()
                triple['subj_id'] = source_id
                triple['predicate'] = "{}.{}".format(str(subject_type).strip(), str(relation).strip())
                triple['predicate_id'] = source_id
                triple['obj_text'] = object_text
                triple['obj_type'] = str(target).strip()
                triple['obj_id'] = target_id
                triple['amr_unknown_var'] = amr_unknown_var
                triple_info.append(triple)

        for frame_id in frame_args:
            frame_roles = frame_args[frame_id]

            if id_to_type[frame_id] == 'have-rel-role-91':
                if 'ARG2' in frame_roles:
                    reified_to_rel[frame_id] = id_to_type[frame_roles['ARG2']]
                elif 'ARG3' in frame_roles:
                    reified_to_rel[frame_id] = id_to_type[frame_roles['ARG3']]
                else:
                    reified_to_rel[frame_id] = 'relation'
                if 'ARG0' in frame_roles and 'ARG1' in frame_roles:
                    for role in ['ARG2', 'ARG3']:
                        if role in frame_roles:
                            del frame_roles[role]

            if id_to_type[frame_id] == 'have-org-role-91':
                if 'ARG2' in frame_roles:
                    reified_to_rel[frame_id] = id_to_type[frame_roles['ARG2']]
                elif 'ARG3' in frame_roles:
                    reified_to_rel[frame_id] = id_to_type[frame_roles['ARG3']]
                else:
                    reified_to_rel[frame_id] = 'position'

                if 'ARG0' in frame_roles and 'ARG1' in frame_roles:
                    for role in ['ARG2', 'ARG3']:
                        if role in frame_roles:
                            del frame_roles[role]

            # logic to handle the special case of frames with a single argument
            if len(frame_roles) == 1:
                frame_roles['unknown'] = "unknown"
                id_to_type['unknown'] = 'unknown'

            rel_keys = sorted(list(frame_roles.keys()))
            for role1, role2 in combinations(rel_keys, 2):
                triple = dict()
                triple['subj_text'] = AMR2Triples.get_variable_text(frame_roles[role1], names, dates)
                triple['subj_type'] = str(id_to_type[frame_roles[role1]]).strip()
                triple['subj_id'] = frame_roles[role1]
                triple['predicate'] = '{}.{}.{}'.format(id_to_type[frame_id], role1.lower(), role2.lower())
                triple['predicate_id'] = frame_id
                triple['obj_text'] = AMR2Triples.get_variable_text(frame_roles[role2], names, dates)
                triple['obj_type'] = str(id_to_type[frame_roles[role2]]).strip()
                triple['obj_id'] = frame_roles[role2]
                triple['amr_unknown_var'] = amr_unknown_var
                triple_info.append(triple)

        return triple_info, names, reified_to_rel, top_node

