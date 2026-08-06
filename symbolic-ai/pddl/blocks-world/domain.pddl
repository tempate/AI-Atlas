(define (domain blocks-world)
    (:requirements :strips)

    (:predicates
        (on ?x ?y)
        (ontable ?x)
        (clear ?x)
        (handempty)
        (holding ?x))

    (:action pick-up
        :parameters (?x)
        :precondition (and (clear ?x) (ontable ?x) (handempty))
        :effect (and (holding ?x)
                     (not (ontable ?x))
                     (not (clear ?x))
                     (not (handempty))))

    (:action put-down
        :parameters (?x)
        :precondition (and (holding ?x))
        :effect (and (ontable ?x)
                     (clear ?x)
                     (handempty)
                     (not (holding ?x))))

    (:action stack
        :parameters (?x ?y)
        :precondition (and (holding ?x) (clear ?y))
        :effect (and (on ?x ?y)
                     (clear ?x)
                     (handempty)
                     (not (clear ?y))
                     (not (holding ?x))))

    (:action unstack
        :parameters (?x ?y)
        :precondition (and (clear ?x) (on ?x ?y) (handempty))
        :effect (and (holding ?x)
                     (clear ?y)
                     (not (clear ?x))
                     (not (handempty))
                     (not (on ?x ?y))))
)
