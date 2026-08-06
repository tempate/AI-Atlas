(define (domain river-crossing)
    (:requirements :strips :equality :typing
                   :negative-preconditions :universal-preconditions)
    (:types item side)

    (:predicates
        (at ?x - item ?s - side)
        (boatat ?s - side)
        (eats ?x - item ?y - item))

    (:action cross-with
        :parameters (?x - item ?fr ?to - side)
        :precondition (and (at ?x ?fr)
                           (boatat ?fr)
                           (not (= ?fr ?to))
                           (forall (?y ?z - item)
                               (not (and (eats ?y ?z)
                                         (at ?y ?fr)
                                         (at ?z ?fr)
                                         (not (= ?y ?x))
                                         (not (= ?z ?x))))))
        :effect (and (at ?x ?to)
                     (boatat ?to)
                     (not (at ?x ?fr))
                     (not (boatat ?fr))))

    (:action cross-alone
        :parameters (?fr ?to - side)
        :precondition (and (boatat ?fr)
                           (not (= ?fr ?to))
                           (forall (?x ?y - item)
                               (not (and (eats ?x ?y)
                                         (at ?x ?fr)
                                         (at ?y ?fr)))))
        :effect (and (boatat ?to)
                     (not (boatat ?fr))))
)
