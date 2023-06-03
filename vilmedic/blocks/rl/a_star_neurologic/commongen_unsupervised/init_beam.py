

common = ['The man', 'A man', 'The boy', 'The girl', 'The woman', 'A person', 'The people', 'The child', 'The kids',
          'The men', 'The boys', 'A woman', 'A boy', 'A girl', 'The person']

character = ['dog', 'kid', 'crowd', 'toddler', 'player', 'worker', 'couple', 'lady', 'team', 'chef', 'band', 'swimmer',
             'surfer', 'referee', 'kids', 'jockey', 'father', 'bird', 'family', 'cat', 'husband',
             'catcher', 'pitcher', 'runner', 'dancer', 'cowboy', 'players', 'photographer', 'bride', 'welder',
             'waiter', 'snowboarder', 'teacher', 'spectator', 'golfer', 'rider', 'professor', 'quarterback',
             'cheerleader', 'individual', 'male', 'puppy', 'policeman', 'artists', 'driver', 'tour',
             'member', 'masseuse', 'musician', 'artist', 'surgeon', 'visitors', 'pedestrian', 'student', 'spectators',
             'congregation', 'tourist', 'trainer', 'scientist', 'greyhound', 'athletes', 'coach',
             'pig', 'performer', 'officer', 'doctor', 'cook', 'fans', 'painter', 'drummer', 'butcher', 'carrier',
             'pelican', 'redheade', 'bulldozer', 'audience', 'diver', 'turker', 'fireman', 'fisherman',
             'biker', 'female', 'skateboarder', 'daughter', 'hiker', 'dogs', 'onlooker', 'soldier', 'pedestrians',
             'workers', 'forecaster', 'dad', 'groomer', 'nurse', 'skier', 'writer', 'robot', 'machine', 'logo'
             'spectator', 'athlete', 'friends']


def add_article(x):
    if x in ['person']:
        return [f'A {x}']

    articles = [f'The {x}']
    if x in ['people', 'crowd', 'congregation', 'audience', 'child'] or x.endswith('s'):
        return articles

    if x[0] in ['a', 'e', 'i', 'o', 'u']:
        articles.append(f'An {x}')
    else:
        articles.append(f'A {x}')
    return articles


def get_init_candidate(constraints_list, beam_size, add_space=False):
    init_candidates = []

    for i, c in enumerate(constraints_list):
        role = []
        keywords = [w.strip() for ci in c for w in ci]

        for r in [x for x in keywords if x in character]:
            role.extend(add_article(r))
        inits = (role + common)[:beam_size]

        if add_space:
            inits = [f' {x}' for x in inits]
        init_candidates.append(inits)

    return init_candidates
