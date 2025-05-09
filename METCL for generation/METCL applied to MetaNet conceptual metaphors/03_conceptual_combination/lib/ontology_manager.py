from owlready2 import *
import os
import types
from random import randint

# class used to build an ontology and check for its consistency
class OntologyManager:

    def __init__ (self, typical_attrs, attrs, line):
        self.attrs = attrs
        self.typical_attrs = typical_attrs
        self.line = line
    
        onto_path.append(os.path.dirname(os.path.abspath(__file__)))
        self.my_world = World()
        self.onto = self.my_world.get_ontology("http://www.example.org/onto.owl#" + str(randint(0, 99)))
                
        self.head = self.create_class('head')
        self.modifier = self.create_class('modifier')
 
        self.add_attrs()
        self.add_typical_attrs()
             
        self.combined = self.create_class('combined')
        self.combined.equivalent_to.append(self.head & self.modifier)

        self.add_typical_combined_attrs()

        self.ind = self.combined("ind")

    # adds rigid properties to the ontology
    def add_attrs(self):
        for property, belongs_to_head in self.attrs:
            # create the attribute, and negate it if it starts with '-'
            tmp_attr = self.create_class(property.replace(' ', '_')
                                            if len(property) > 0 and property[0] != '-' 
                                            else property[1:].replace(' ', '_') )
            
            if len(property) > 0 and property[0] == '-':
                tmp_attr = Not(tmp_attr)            

            # add the property to the corresponding concept
            if belongs_to_head:
                self.head.is_a.append(tmp_attr)
            else:
                self.modifier.is_a.append(tmp_attr) 

    # adds typical properties to the ontology
    def add_typical_attrs(self):

        for property, probability, belongs_to_head in self.typical_attrs:
            # create the attribute, and negate it if it starts with '-'
            tmp_attr = self.create_class(property.replace(' ', '_')
                                            if property[0] != '-'
                                            else property[1:].replace(' ', '_'))
            
            if property[0] == '-':
                tmp_attr = Not(tmp_attr)

            # add the property to the corresponding concept
            # (adding all the relations implied by typicality)
            if belongs_to_head:
                head1 = self.create_class('head1')
                heads = self.create_class('heads')
                not_head1 = self.create_class('not_head1')

                heads.equivalent_to.append(self.head & head1)
                not_head1.equivalent_to.append(Not(head1))
                head_r = self.create_property('head_R')
                
                heads.is_a.append(tmp_attr)
                head1.is_a.append(head_r.only(Not(self.head) & head1))
                not_head1.is_a.append(head_r.some(self.head & head1))
                
            else:
                modifier1 = self.create_class('modifier1')
                modifiers = self.create_class('modifiers')
                not_modifier1 = self.create_class('not_modifier1')

                modifiers.equivalent_to.append(self.modifier & modifier1)
                not_modifier1.equivalent_to.append(Not(modifier1))
                modifier_r = self.create_property('modifier_R')
                
                modifiers.is_a.append(tmp_attr)
                modifier1.is_a.append(modifier_r.only(
                                    Not(self.modifier) & modifier1) )
                not_modifier1.is_a.append(modifier_r.some(
                                    self.modifier & modifier1) )

    # create combined concept in the ontology
    def add_typical_combined_attrs(self):
        for i in range(len(self.typical_attrs)):
            if self.line[i] == 1:
                property, probability, belongs_to_head = self.typical_attrs[i]
                tmp_attr = self.create_class(property.replace(' ', '_')
                                                if property[0] != '-'
                                                else property[1:].replace(' ', '_'))
                
                if property[0] == '-':
                    tmp_attr = Not(tmp_attr)

                combined1 = self.create_class('combined1')
                combineds = self.create_class('combineds')
                not_combined1 = self.create_class('not_combined1')

                combineds.equivalent_to.append(self.combined & combined1)
                not_combined1.equivalent_to.append(Not(combined1))
                combined_r = self.create_property('combined_R')
                
                combineds.is_a.append(tmp_attr)
                combined1.is_a.append(combined_r.only(Not(self.combined) & combined1))
                not_combined1.is_a.append(combined_r.some(self.combined & combined1))
    
    # dynamically creates ontology classes
    def create_class(self, name, parent = Thing):
        with self.onto:
            new_class = types.new_class(name, (parent,))
        return new_class

    # dynamically creates ontology properties
    def create_property(self, name):
        with self.onto:
            new_prop = types.new_class(name, (ObjectProperty,))
        return new_prop

    # call HermiT reasoner to check the consistency of the ontology
    def is_consistent(self):        
        try:
            with self.onto:
                sync_reasoner(self.my_world, debug = 0)
        except subprocess.CalledProcessError as inst:
            return False
        except Exception as e:
            print(f"Unexpeced error during reasoning: {e}")
            return False
        # if no exceptions are raised, the ontology is consistent
        return True    

