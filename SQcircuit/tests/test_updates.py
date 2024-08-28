import SQcircuit as sq

def test_elem_update():
    sq.log_to_stdout()

    C = sq.Capacitor(5, 'GHz')
    JJ = sq.Junction(3, 'GHz')

    cr = sq.Circuit({(0, 1): [C, JJ]})
    cr.set_trunc_nums([50])

    print('Updating elements')
    C.set_value(1, 'GHz')
    JJ.set_value(10, 'GHz')
    print('Calling diag.')
    cr.diag(10) # <- works when you call functions
    print('Updating junction')
    C.set_value(10, 'GHz')
    print('Referencing capacitance matrix')
    print(cr.C) # <- or even just access normal attributes
