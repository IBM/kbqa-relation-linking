import json
import re
import difflib

from relation_linking_core.metadata_generator.amr_graph_to_triples import AMR2Triples
from relation_linking_core.metadata_generator.amr_types import AmrTypes


class EntityUtils:

    @classmethod
    def clean_string(cls, str_value):
        if str_value:
            return str(str_value).replace('\'', '').replace('"', '')

    @classmethod
    def get_normalized_term(cls, surface_form):
        surface_form_splits = surface_form.split()
        if len(surface_form_splits) == 1:
            lemma = AMR2Triples.lemmatizer.lemmatize(surface_form_splits[0])
        elif len(surface_form_splits) > 1:
            prefix = surface_form_splits[:-1]
            lemma = AMR2Triples.lemmatizer.lemmatize(surface_form_splits[-1])
            prefix.append(lemma)
            lemma = " ".join(prefix)
        else:
            lemma = surface_form
        return lemma

    @classmethod
    def align_entities(cls, amr_nodes, entities):
        amr_entity_alignments = dict()
        normalized_to_surface_form = dict()
        for surface_form in entities:
            if surface_form in amr_nodes:
                amr_entity_alignments[surface_form] = entities[surface_form]['uri']
            if surface_form.lower() in amr_nodes:
                amr_entity_alignments[surface_form.lower()] = entities[surface_form]['uri']

            normalized_form = EntityUtils.get_normalized_term(surface_form)
            normalized_to_surface_form[normalized_form] = surface_form
            if normalized_form in amr_nodes:
                amr_entity_alignments[normalized_form] = entities[surface_form]['uri']
        return amr_entity_alignments, normalized_to_surface_form

    @classmethod
    def align_entities_annotated(cls, text, amr_nodes, entites):
        entity_map = EntityUtils.get_entity_annotation_map()
        amr_nodes_copy = list(amr_nodes)

        amr_entity_alignments = dict()
        normalized_to_surface_form = dict()

        entities = entity_map[text]

        for entity in entities:
            close_match = difflib.get_close_matches(entity, amr_nodes_copy, cutoff=0.2)
            if close_match:
                amr_entity_alignments[close_match[0]] = entities[entity]
                normalized_to_surface_form[close_match[0]] = entity
                amr_nodes_copy.remove(close_match[0])

        return amr_entity_alignments, normalized_to_surface_form

    @classmethod
    def get_entities(cls, amr_tree):
        """ extract the entities from the extended AMR tree."""
        temp_entities = dict()
        for head, rel, tail in amr_tree.triples():
            head = EntityUtils.clean_string(head)
            tail = EntityUtils.clean_string(tail)
            if rel in ['surface_form', 'uri']:
                entity = temp_entities.get(head, dict())
                entity[rel] = tail.lower() if rel == 'surface_form' else tail
                temp_entities[head] = entity
            elif 'type' == rel:
                entity = temp_entities.get(head, dict())
                types = entity.get('types', set())
                types.add(tail)
                entity['types'] = types
                temp_entities[head] = entity
        entities = dict()
        for entity in temp_entities.values():
            if 'surface_form' in entity:
                entities[entity['surface_form']] = entity
        return entities

    @classmethod
    def link_entities_types(cls, triple, amr_entity_alignments, answer_types):
        subj_uri, obj_uri, subj_type_uri, obj_type_uri, answer_type = None, None, None, None, None

        if len(answer_types) > 0 and answer_types[0][0].startswith('http://dbpedia.org/ontology/'):
            answer_type = answer_types[0][0]

        subj_text = triple['subj_text'].lower()
        subj_type = triple['subj_type'].lower()
        obj_text = triple['obj_text'].lower()
        obj_type = triple['obj_type'].lower()

        if subj_text in amr_entity_alignments:
            if amr_entity_alignments[subj_text].startswith('http://dbpedia.org/resource/'):
                subj_uri = amr_entity_alignments[subj_text]
            elif amr_entity_alignments[subj_text].startswith('http://dbpedia.org/ontology/'):
                subj_type_uri = amr_entity_alignments[subj_text]
        if subj_type in amr_entity_alignments:
            if amr_entity_alignments[subj_type].startswith('http://dbpedia.org/resource/'):
                subj_uri = amr_entity_alignments[subj_type]
            elif amr_entity_alignments[subj_type].startswith('http://dbpedia.org/ontology/'):
                subj_type_uri = amr_entity_alignments[subj_type]

        if not subj_type_uri:
            if triple['subj_type'] in AmrTypes.dbpedia and AmrTypes.dbpedia[triple['subj_type']] != "owl:Thing":
                subj_type_uri = AmrTypes.dbpedia[triple['subj_type']]
        if not subj_type_uri and triple['amr_unknown_var'] == triple['subj_id'] and answer_type:
            subj_type_uri = answer_type

        if obj_text in amr_entity_alignments:
            if amr_entity_alignments[obj_text].startswith('http://dbpedia.org/resource/'):
                obj_uri = amr_entity_alignments[obj_text]
            elif amr_entity_alignments[obj_text].startswith('http://dbpedia.org/ontology/'):
                obj_type_uri = amr_entity_alignments[obj_text]
        if obj_type in amr_entity_alignments:
            if amr_entity_alignments[obj_type].startswith('http://dbpedia.org/resource/'):
                obj_uri = amr_entity_alignments[obj_type]
            elif amr_entity_alignments[obj_type].startswith('http://dbpedia.org/ontology/'):
                obj_type_uri = amr_entity_alignments[obj_type]

        if not obj_type_uri:
            if triple['obj_type'] in AmrTypes.dbpedia and AmrTypes.dbpedia[triple['obj_type']] != "owl:Thing":
                obj_type_uri = AmrTypes.dbpedia[triple['obj_type']]
        if not obj_type_uri and triple['amr_unknown_var'] == triple['obj_id'] and answer_type:
            obj_type_uri = answer_type

        if subj_type_uri:
            subj_type_uri = subj_type_uri.replace("http://dbpedia.org/ontology/", "dbo:")
        if obj_type_uri:
            obj_type_uri = obj_type_uri.replace("http://dbpedia.org/ontology/", "dbo:")

        triple['subj_uri'] = subj_uri
        triple['subj_type_uri'] = subj_type_uri
        triple['obj_uri'] = obj_uri
        triple['obj_type_uri'] = obj_type_uri


    @classmethod
    def get_entity_annotation_map(cls):
        path = "/Users/nandana.sampath.mihindukulasooriya@ibm.com/Downloads/FullyAnnotated_LCQuAD5000left.json"
        with open(path) as json_file:
            en_data = json.load(json_file)

        q_to_entities = dict()

        for question in en_data:
            text = question['question']
            entities = dict()
            for entity_mapping in question['entity mapping']:
                entities[entity_mapping['label']] = entity_mapping['uri']

            for predicate_mapping in question['predicate mapping']:
                uri = predicate_mapping['uri']
                if 'uri' not in predicate_mapping or 'label' not in predicate_mapping:
                    continue

                if uri.startswith("http://dbpedia.org/resource/") or re.match('http://dbpedia.org/ontology/[A-Z]', uri):
                    entities[predicate_mapping['label']] = uri

            q_to_entities[text] = entities

        return q_to_entities









