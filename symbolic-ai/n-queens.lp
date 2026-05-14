% Input queens, rows, and columns
queen(1..n).
row(1..n).
col(1..n).

% Each queen is exactly in one row and one column.
{ set_pos(Q, R, C) : row(R), col(C) } = 1 :- queen(Q).

% Two distinct queens cannot be in the same row.
:- set_pos(Q1, R, C1), set_pos(Q2, R, C2), Q1 < Q2.

% Two distinct queens cannot be in the same column.
:- set_pos(Q1, R1, C), set_pos(Q2, R2, C), Q1 < Q2.

% Two distinct queens cannot be in the same diagonal.
:- set_pos(Q1, R1, C1), set_pos(Q2, R2, C2), Q1 < Q2,
   R1 - C1 = R2 - C2.

% Two distinct queens cannot be in the same anti-diagonal.
:- set_pos(Q1, R1, C1), set_pos(Q2, R2, C2), Q1 < Q2,
   R1 + C1 = R2 + C2.

#show set_pos/3.
