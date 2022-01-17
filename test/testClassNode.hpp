#ifndef testClassNode_h
#define testClassNode_h

#include <memory>

class Link;


class Node
{
    private:
        int id;
        std::shared_ptr<Link> prev_link;
        std::shared_ptr<Link> next_link;
    
    public:
        Node(int id) : id(id) {};
        ~Node(){};

        int getId();
        std::shared_ptr<Link> getPrevLink();
        std::shared_ptr<Link> getNextLink();

        void setPrevLink(std::shared_ptr<Link> link);
        void setNextLink(std::shared_ptr<Link> link);
};


#endif