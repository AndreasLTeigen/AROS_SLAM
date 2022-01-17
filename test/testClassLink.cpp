#include <memory>

#include "testClassLink.hpp"

int Link::getId()
{
    return this->id;
}

std::shared_ptr<Node> Link::getPrevNode()
{
    return this->prev_node.lock();
}
std::shared_ptr<Node> Link::getNextNode()
{
    return this->next_node.lock();
}

void Link::setPrevNode(std::shared_ptr<Node> node)
{
    this->prev_node = node;
}
void Link::setNextNode(std::shared_ptr<Node> node)
{
    this->next_node = node;
}