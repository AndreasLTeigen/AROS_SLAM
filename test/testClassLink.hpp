
#ifndef testClassLink_h
#define testClassLink_h

#include <memory>

class Node;

class Link
{
    private:
        int id;
        std::weak_ptr<Node> prev_node;
        std::weak_ptr<Node> next_node;
    
    public:
        Link(int id) : id(id) {};
        ~Link(){};

        int getId();
        std::shared_ptr<Node> getPrevNode();
        std::shared_ptr<Node> getNextNode();

        void setPrevNode(std::shared_ptr<Node> node);
        void setNextNode(std::shared_ptr<Node> node);
};

#endif