#!/usr/bin/env python

import xml.etree.ElementTree
from conf import breathe_default_project, breathe_projects

doxyxml_dir = breathe_projects[breathe_default_project]
index_xml = '{}/index.xml'.format(doxyxml_dir)

tree = xml.etree.ElementTree.parse(index_xml).getroot()

with open('Structs.rst', 'w') as file:
    file.write('===========\n')
    file.write('Structures\n')
    file.write('===========\n')
    file.write('\n')

    for node in tree.findall('compound'):
        if node.attrib['kind'] == 'struct':
            name = 'unknown'
            members = []

            for child in list(node):
                if child.tag == 'name':
                    name = child.text
                elif child.tag == 'member':
                    for member in list(child):
                        members.append(member.text)

            file.write('.. doxygenstruct:: {}\n'.format(name))
            file.write('   :project: {}\n'.format(breathe_default_project))
            file.write('   :members: \n')
            file.write('\n')
