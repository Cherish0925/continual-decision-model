"""Factorized request domains shared by language and learned planners."""
from cdm.digital import NativeCandidate, _objects, _object_text, _text, _signed_log, observed_numbers


def request_candidates(observation, prefix=None):
    """Select operation, then its arguments, without a quadratic pair list.

    Arguments bind only previously observed objects or documented non-object
    domains. Accessibility affects execution cost, not eligibility for a request.
    A call returns the next field's alternatives; it never updates model state.
    """
    prefix = dict(prefix or {})
    ui, objects = observation['ui'], _objects(observation)
    dialog = ui.get('dialog_box', {})
    candidates = []

    def add(action, text, first=None, second=None):
        a,b=objects.get(first,{}),objects.get(second,{})
        numbers=observed_numbers(text)
        numeric=(float(a.get('inventory',False)),float(a.get('accessible',False)),
                 float(b.get('inventory',False)),float(b.get('accessible',False)),
                 _signed_log(float(a.get('distance',0))),_signed_log(float(b.get('distance',0))),
                 _signed_log(numbers[0]) if numbers else 0.,float(bool(numbers)))
        candidates.append(NativeCandidate(action,_text(text),numeric,(first,second)))

    if dialog.get('is_in_dialog') and dialog.get('dialogOptions'):
        if prefix:
            return []
        options=dialog['dialogOptions']
        for key,text in (options.items() if isinstance(options,dict) else enumerate(options,1)):
            add({'chosen_dialog_option_int':int(key)},'DIALOG: '+str(text))
    elif not prefix:
        for op,description in sorted(observation.get('actions',{}).items()):
            args=description.get('args',[])
            if args and not request_candidates(observation,{'action':op}):
                continue
            add({'action':op},op+': '+_text(description.get('desc','')))
    else:
        op=prefix['action'];args=observation['actions'][op].get('args',[])
        remaining=[a for a in args if a not in prefix]
        if not remaining:
            return []
        key=remaining[0]
        if op in ('MOVE_DIRECTION','ROTATE_DIRECTION'):
            domain=['north','east','south','west']
        elif op=='TELEPORT_TO_LOCATION':
            domain=list(observation.get('teleport_locations',{}))
        elif op=='DISCOVERY_FEED_GET_POST_BY_ID':
            domain=list(dict.fromkeys(p['postID'] for rows in ui.get('discoveryFeed',{}).values()
                if isinstance(rows,list) for p in rows if isinstance(p,dict) and 'postID' in p))
        else:
            domain=[h for h in objects if h!=prefix.get('arg1')]
        for value in domain:
            action={**prefix,key:value}
            first=action.get('arg1') if action.get('arg1') in objects else None
            second=action.get('arg2') if action.get('arg2') in objects else None
            parts=[op]
            for argument in args:
                if argument in action:
                    v=action[argument]
                    parts.append(_object_text(objects[v]) if v in objects else str(v))
            add(action,' -> '.join(parts),first,second)
    return sorted(candidates,key=lambda c:(c.text,c.numeric))


def request_stages(observation, action):
    """Teacher-forced argument stages; only the completed request is executed."""
    prefix={};stages=[]
    for _ in range(3):
        choices=request_candidates(observation,prefix)
        if not choices:
            break
        matching=[i for i,c in enumerate(choices) if all(action.get(k)==v for k,v in c.action.items())]
        if len(matching)!=1:
            raise ValueError('The recorded request does not have one public argument continuation')
        index=matching[0];stages.append((choices,index));prefix=choices[index].action
    if prefix!=action:
        raise ValueError('The recorded request contains fields outside its public domain')
    return stages
