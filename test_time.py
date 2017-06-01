import time
import sys

global_timer_name = 'tt'


def tic(timer_name=''):
    if not global_timer_name in globals():
        globals()[global_timer_name] = {}

    timer_dict = globals()[global_timer_name]
    if not timer_name in timer_dict:
        timer_dict[timer_name] = {'total_elapsed': 0, 'last_elapsed': 0}
    timer_dict[timer_name]['last_clock'] = time.time()


def toc(timer_name=''):
    if not global_timer_name in globals():
        print 'not tic before'
        return -1
    timer_dict = globals()[global_timer_name]
    if not timer_name in timer_dict:
        print 'not tic before'
        return -1
    else:
        cur_clock = time.time()
        elapsed = cur_clock - timer_dict[timer_name]['last_clock']
        timer_dict[timer_name]['total_elapsed'] += elapsed
        timer_dict[timer_name]['last_clock'] = cur_clock
        timer_dict[timer_name]['last_elapsed'] = elapsed
        #total_elapsed = sum(timer_dict[timer_name]['elapsed'])
        return elapsed
        #print '[DEBUG]' + timer_name + ' elapsed time %.8f seconds, total elapsed %.4f seconds' % (elapsed, total_elapsed)
    #sys.stdout.flush()


def time_elapsed(timer_name=''):
    if not global_timer_name in globals():
        print 'not tic before'
        return 0
    timer_dict = globals()[global_timer_name]
    if not timer_name in timer_dict:
        print 'not tic before'
        return 0
    return timer_dict[timer_name]['total_elapsed']


def get_last_time_elapsed(timer_name=''):
    if not global_timer_name in globals():
        print 'not tic before'
        return 0
    timer_dict = globals()[global_timer_name]
    return timer_dict[timer_name]['last_elapsed']