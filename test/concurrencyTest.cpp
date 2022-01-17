#include <iostream>
#include <vector>
#include <memory>
#include <map>

#include "concurrencyTest.hpp"
#include "testClassNode.hpp"
#include "testClassLink.hpp"


void vectorItterationTest()
{
    std::vector<int> vect{1,2,3,4,5,6,7,8,9};

    for (int x: getVector())
    {
        std::cout << x << std::endl;
    }
}


void sharePointerCounterTest()
{
    int n_nodes = 20;
    int n_links = n_nodes - 1;

    // Initializing the object arrays
    std::vector<std::shared_ptr<Node>> node_array;
    std::vector<std::shared_ptr<Link>> link_array;

    for (int i = 0; i < n_nodes; i++)
    {
        node_array.push_back(std::make_shared<Node>(i));
    }
    for (int i = 0; i < n_links; i++)
    {
        link_array.push_back(std::make_shared<Link>(i));
    }

    // Interlieving nodes and links

    node_array[0]->setNextLink(link_array[0]);
    node_array[node_array.size()-1]->setPrevLink(link_array[link_array.size()-1]);
    link_array[0]->setPrevNode(node_array[0]);
    link_array[link_array.size()-1]->setNextNode(node_array[node_array.size()-1]);
    for (int i = 1; i < node_array.size()-1; i++)
    {
        node_array[i]->setPrevLink(link_array[i-1]);
        node_array[i]->setNextLink(link_array[i]);
        link_array[i-1]->setNextNode(node_array[i]);
        link_array[i]->setPrevNode(node_array[i]);
    }

    int j = 1;
    //std::shared_ptr<Node> node = node_array[j];
    //std::shared_ptr<Link> link = link_array[j];
    std::shared_ptr<Node> saved_node = node_array[0]->getNextLink()->getNextNode()->getNextLink()->getNextNode();

    std::cout << "Node use count: " << node_array[j].use_count() << std::endl;
    std::cout << "Link use count: " << link_array[j].use_count() << std::endl;




    /*
    std::shared_ptr<Node> node_it = node_array[node_array.size()-1];
    std::shared_ptr<Link> link_it;
    std::cout << ".............." << std::endl;
    while (node_it->getId() != 0)
    {
        link_it = node_it->getPrevLink();
        node_it = link_it->getPrevNode();
        std::cout << node_it->getId() << std::endl;
    }*/
}

void emptyMapTest()
{
    std::vector vect = getVectorFromMap(1);
    for ( int value : vect )
    {
        std::cout << value << std::endl;
    }
}


// Helper functions

std::vector<int> getVector()
{
    std::cout << "Running <getVector>.." << std::endl;
    std::vector<int> vect{1,2,3,4,5,6,7,8,9};
    return vect;
}

std::vector<int> getVectorFromMap(int key)
{
    std::map<int, std::vector<int>> ex_map;
    std::vector<int> fill_vec {2,5,1,4,2,3};
    ex_map[0] = fill_vec;
    return ex_map[key];
}