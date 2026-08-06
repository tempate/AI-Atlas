% Input graph
node(1..5).
edge(1,2). edge(2,3). edge(3,4). edge(4,5). edge(5,1). edge(1,3).

% Choose exactly one color per node
{ color(N,red); color(N,green); color(N,blue) } = 1 :- node(N).

% No two adjacent graphs can have the same color
:- color(Node1,Color), color(Node2,Color), edge(Node1,Node2).

#show color/2.
