#include <memory>

#include "testClassNode.hpp"

int Node::getId()
{
    return this->id;
}

std::shared_ptr<Link> Node::getPrevLink()
{
    return this->prev_link;
}
std::shared_ptr<Link> Node::getNextLink()
{
    return this->next_link;
}

void Node::setPrevLink(std::shared_ptr<Link> link)
{
    this->prev_link = link;
}
void Node::setNextLink(std::shared_ptr<Link> link)
{
    this->next_link = link;
}