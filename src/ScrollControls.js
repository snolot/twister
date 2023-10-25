import {
	Raycaster,
	Vector2,
	MathUtils,
} from '../libs/build/three.module.js';

const ScrollControls = (renderer, _options) => {
	const options = {
		eps: 0.00001,
	  	enabled: true,
	  	infinite: false,
	  	horizontal: false,
	  	pages: 1,
	  	distance: 1,
	  	damping: 4,
	  	style: {},
	  	children: [],
	  	group: null,
	  	onScrolled: (value) => {
	  		//console.log(value);
	  	}
	};

	Object.assign(options, _options);

	const state = {
		el: document.createElement('div'),
		eps: options.eps,
	  	fill: document.createElement('div'),
	  	fixed: document.createElement('div'),
	  	horizontal: options.horizontal,
	  	damping: options.damping,
	 	offset: 0,
	  	delta: 0,
	  	pages: options.pages,
	  	size:{
	  		width:renderer.domElement.width, 
	  		height: renderer.domElement.height
	  	},
	};

	const target = renderer.domElement.parentNode;
	let scroll = null;
	const raycaster = new Raycaster();
	const pointer = new Vector2();

    let current = 0
    let _disableScroll = true
    let firstRun = true

    let _enabled = true;
    
    let ts   = Date.now();
	let yPos = 0;
	let _scrollVelocity=0;

	const range = (from, distance, margin = 0) => {
		let offset = state.offset;

		//console.log(offset)
        const start = from - margin;
        const end = start + distance + margin * 2;

        return offset < start ? 0 : offset > end ? 1 : (offset - start) / (end - start)
    };
    
    // 0-1-0 for a range between from -> from + distance
    const curve = (from, distance, margin = 0) => {
        return Math.sin(range(from, distance, margin) * Math.PI);
    };

    // true/false for a range between from -> from + distance
    const visible = (from, distance, margin = 0) => {
    	const {offset} = state;

        const start = from - margin;
        const end = start + distance + margin * 2;
        //console.log(offset)
        return offset >= start && offset <= end;
    };

   	const computeOffsets = ({ clientX, clientY }) => {
	    pointer.set(
	      	clientX - target.offsetLeft,
	      	clientY - target.offsetTop
	    );
	};

	const onPointerMove = ( event ) => {
		// calculate pointer position in normalized device coordinates
		// (-1 to +1) for both components
		pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
		pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

		//console.log(pointer);
	};

    const init = () => {
    	const { el, fixed, fill, horizontal, pages } = state;
    	const { distance } = options;

    	el.style.position = 'absolute';
	    el.style.width = '100%';
	    el.style.height = '100%';
	    el.style[horizontal ? 'overflowX' : 'overflowY'] = 'auto';
	    el.style[horizontal ? 'overflowY' : 'overflowX'] = 'hidden';
	    el.style.top = '0px';
	    el.style.left = '0px';

	    for (const key in options.style) {
	      	el.style[key] = style[key];
	    }

	    fixed.style.position = 'sticky';
	    fixed.style.top = '0px';
	    fixed.style.left = '0px';
	    fixed.style.width = '100%';
	    fixed.style.height = '100%';
	    fixed.style.overflow = 'hidden';
	    el.appendChild(fixed);

	    fill.style.height = horizontal ? '100%' : `${pages * distance * 100}%`;
	    fill.style.width = horizontal ? `${pages * distance * 100}%` : '100%';
	    fill.style.pointerEvents = 'none';
	    el.appendChild(fill);
	    target.appendChild(el);

	    // Init scroll one pixel in to allow upward/leftward scroll
	    state.el[horizontal ? 'scrollLeft' : 'scrollTop'] = 1

	    window.addEventListener('pointermove', onPointerMove);
	    
    };

    const onScroll = () => {
      	// Prevent first scroll because it is indirectly caused by the one pixel offset
      	if (!options.enabled || firstRun) return
      
      	current = state.el[state.horizontal ? 'scrollLeft' : 'scrollTop'];
      	scroll = current / scrollThreshold;

      	let _ts   = Date.now();
  		let _yPos = scroll;

		// calculate the velocity as change in vertical position
		// over change in milliseconds
		_scrollVelocity = (_yPos - yPos) / (_ts - ts) * 100;

  		//console.log('velocity: ', _scrollVelocity);

		// update "prior" values
		ts   = _ts;
		yPos = _yPos;

      	if (options.infinite) {
        	if (!_disableScroll) {
          		if (current >= scrollThreshold) {
            		const damp = 1 - state.offset;
		            state.el[state.horizontal ? 'scrollLeft' : 'scrollTop'] = 1;
		            scroll = state.offset = -damp;
		            _disableScroll = true;
          		} else if (current <= 0) {
		            const damp = 1 + state.offset;
		            state.el[state.horizontal ? 'scrollLeft' : 'scrollTop'] = scrollLength;
		            scroll = state.offset = damp;
		            _disableScroll = true;
          		}
        	}

        	if(_disableScroll)
        		setTimeout(() => (_disableScroll = false), 40)
      	}
    }

    state.el.addEventListener('scroll', onScroll, { passive: true })

    let last = 0;

    const onWheel = (e) => (state.el.scrollLeft += e.deltaY / 2);

    if (state.horizontal)
    	state.el.addEventListener('wheel', onWheel, { passive: true })

    const {group} = options;

    const update = (dt) => {
    	firstRun = false

    	const {width, height} = state.size;
    	const {damping, pages, horizontal} = state;
    	let delta = state.delta;
    	let offset = state.offset;

    	last = offset;

    	state.offset = damp(last, scroll, damping, dt)    	
    	state.delta = damp(delta, Math.abs(last - offset), damping, dt)

    	_scrollVelocity *= .975;

    	if(group){
    		group.position.x = horizontal ? -width * (pages - 1) * offset : 0
    		group.position.y = horizontal ? 0 : height * (pages - 1) * offset
    	}	
    	console.log(scroll)
    	options.onScrolled(scroll);
    };

    const lockScroll = (isLock) => {
    	if(!isLock){

    	}else{

    	}
    }

    init();

    const containerLength = state.size[state.horizontal ? 'width' : 'height']
    const scrollLength = state.el[state.horizontal ? 'scrollWidth' : 'scrollHeight']
    const scrollThreshold = scrollLength - containerLength
    const damp = MathUtils.damp;

    //	console.log(scrollLength, containerLength, scrollThreshold);

	const base = {
		range,
		curve,
		visible,
		update,
	};

	Object.defineProperty(base, 'mouse', {
		get: () => pointer,
	});

	Object.defineProperty(base, 'scrollVelocity', {
		get: () => _scrollVelocity,
	});

	Object.defineProperty(base, 'disableScroll', {
		get: () => _disableScroll,
		set:(value) => _disableScroll = value
	});

	let storeScrollTop = 0;

	Object.defineProperty(base, 'enabled', {
		get: () => _enabled,
		set: (value) => {
			if(value === true){
				//state.el.scrollTop = storeScrollTop;
				state.el.addEventListener('scroll', onScroll, { passive: true });
				state.el.style.overflowY = 'auto';
			} else {
				state.el.removeEventListener('scroll', onScroll);
				state.el.style.overflowY = 'hidden';
				//storeScrollTop = state.el[state.horizontal ? 'scrollLeft' : 'scrollTop'];
				_scrollVelocity = 0;
			}
			_enabled = value;
		}
	})

	return base;
};

export default ScrollControls;
