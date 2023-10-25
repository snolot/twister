import {
	AdditiveBlending,
	BoxBufferGeometry,
	Camera,
	ClampToEdgeWrapping,
	Color,
	DataTexture,
	ExtrudeGeometry,
	FloatType,
	InstancedBufferAttribute,
	InstancedBufferGeometry,
	MathUtils,
	Mesh,
	MeshDepthMaterial,
	MeshStandardMaterial,
	NearestFilter,
	PlaneBufferGeometry,
	RGBADepthPacking,
	RGBAFormat,
	Scene,
	ShaderMaterial,
	Shape,
	SphereBufferGeometry,
	TetrahedronBufferGeometry,
	TextureLoader,
	Vector2,
	Vector3,
	WebGLRenderTarget,
} from '../libs/build/three.module.js';

import {GPUComputationRenderer} from '../libs/examples/jsm/misc/GPUComputationRenderer.js';

const Flocking = (renderer, scene, camera, _options) => {
	const options = {
		touch: false,
	};

	const computeVelShader = `
		
		uniform float uRange;
		uniform float uRangeTop;

		uniform float uRadius;

		uniform float uTime;
		uniform sampler2D textureRandom;
		uniform sampler2D defaultPosTexture;
		uniform float uNoiseScale;
		uniform float uMaxRadius;
		uniform vec3 uHit;
		uniform float uIsMouseDown;
		uniform float uSpeed;

		vec4 permute(vec4 x) { return mod(((x*34.00)+1.00)*x, 289.00); }
		vec4 taylorInvSqrt(vec4 r) { return 1.79 - 0.85 * r; }

		float snoise(vec3 v){
			const vec2 C = vec2(1.00/6.00, 1.00/3.00) ;
			const vec4 D = vec4(0.00, 0.50, 1.00, 2.00);
			
			vec3 i = floor(v + dot(v, C.yyy) );
			vec3 x0 = v - i + dot(i, C.xxx) ;
			
			vec3 g = step(x0.yzx, x0.xyz);
			vec3 l = 1.00 - g;
			vec3 i1 = min( g.xyz, l.zxy );
			vec3 i2 = max( g.xyz, l.zxy );
			
			vec3 x1 = x0 - i1 + 1.00 * C.xxx;
			vec3 x2 = x0 - i2 + 2.00 * C.xxx;
			vec3 x3 = x0 - 1. + 3.00 * C.xxx;
			
			i = mod(i, 289.00 );
			vec4 p = permute( permute( permute( i.z + vec4(0.00, i1.z, i2.z, 1.00 )) + i.y + vec4(0.00, i1.y, i2.y, 1.00 )) + i.x + vec4(0.00, i1.x, i2.x, 1.00 ));
			
			float n_ = 1.00/7.00;
			vec3 ns = n_ * D.wyz - D.xzx;
			
			vec4 j = p - 49.00 * floor(p * ns.z *ns.z);
			
			vec4 x_ = floor(j * ns.z);
			vec4 y_ = floor(j - 7.00 * x_ );
			
			vec4 x = x_ *ns.x + ns.yyyy;
			vec4 y = y_ *ns.x + ns.yyyy;
			vec4 h = 1.00 - abs(x) - abs(y);
			
			vec4 b0 = vec4( x.xy, y.xy );
			vec4 b1 = vec4( x.zw, y.zw );
			
			vec4 s0 = floor(b0)*2.00 + 1.00;
			vec4 s1 = floor(b1)*2.00 + 1.00;
			vec4 sh = -step(h, vec4(0.00));
			
			vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
			vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
			
			vec3 p0 = vec3(a0.xy,h.x);
			vec3 p1 = vec3(a0.zw,h.y);
			vec3 p2 = vec3(a1.xy,h.z);
			vec3 p3 = vec3(a1.zw,h.w);
			
			vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
			p0 *= norm.x;
			p1 *= norm.y;
			p2 *= norm.z;
			p3 *= norm.w;
			
			vec4 m = max(0.60 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.00);
			m = m * m;
			return 42.00 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
		}

		float snoise(float x, float y, float z){
			return snoise(vec3(x, y, z));
		}

		float rand(vec2 co){
		    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
		}

		vec3 getPosition(vec3 value) {
			vec3 pos;

			pos.y = value.y;
			pos.x = cos(value.z) * value.x;
			pos.z = sin(value.z) * value.x;
			return pos;
		}

		const float PI = 3.141592657;

		const float mouseRadius = 1.0;

		//const float range = 250.0;
		const float mixture = .5;
		//const float radius = 400.0;
		const float minRadius = 1.0;

		float cubicIn(float t) {
		  return t * t * t;
		}

		float exponentialIn(float t) {
		  return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0));
		}

		float exponentialOut(float t) {
		  return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t);
		}

		float getDelta(float s, float e, float v) {
			return (v - s) / (e - s);
		}

		const float lowThreshold = 10.;

		void main(void) {

			vec2 uv = gl_FragCoord.xy / resolution.xy;
	
			vec2 uvPos      = uv;
			vec2 uvExtra    = uv;
			vec3 orgPos 	= texture2D(texturePosition, uv).rgb;
			vec3 vel 		= texture2D(textureVelocity, uv).rgb;
			vec3 extra 		= texture2D(textureRandom, uv).rgb;
			vec3 pos 		= getPosition(orgPos);

			float yOffset 	=  (pos.y + uRangeTop) / (uRangeTop * 2.0);
			
			const float posOffset = .01;
			const float mixOffset = .95;
			float aRotation = .0005 * mix(extra.x, 1.0, mixOffset);
			float aRadius   = .01 * mix(extra.y, 1.0, mixOffset);
			float aY 		= .005 * mix(extra.z, 1.0, mixOffset) + cubicIn(1.0-yOffset) * .05;
			
			float ax 		= snoise(pos.x*posOffset+uTime, pos.y*posOffset+uTime, pos.z*posOffset+uTime) * aRadius;
			float ay 		= (snoise(pos.y*posOffset+uTime, pos.z*posOffset+uTime, pos.x*posOffset+uTime) + .85) * aY;
			float az 		= (snoise(pos.z*posOffset+uTime, pos.x*posOffset+uTime, pos.y*posOffset+uTime) + .85) * aRotation;

			vec3 acc = vec3(0.0);

			vel += vec3(ax, -ay, az);
			float minRadius = 10.0;

			// float ty = 
			float maxRadius = uRadius * exponentialIn(mix(yOffset, 1.0, .2));
			
			float dist = distance(uHit, orgPos);
			float p = dist / mouseRadius;
			vec3 dir = normalize(orgPos - pos);

			vel += acc;

			if(orgPos.x < minRadius) {
				vel.x += 1.0/(orgPos.x/minRadius) * .01;
			} else if(orgPos.x > maxRadius) {
				vel.x -= (orgPos.x - maxRadius) * .00015;
			}

			const float maxRotationSpeed = 5.;
			if(vel.z > maxRotationSpeed) {
				vel.z -= (vel.z - maxRotationSpeed) * .1;
			}

			
			//	DECREASE
			vel *= .975;
			gl_FragColor = vec4(vel, 1.0);	
		}
	`;

	const computePosShader =  `
		
		uniform float uTime;
		uniform float uRange;
		uniform float uRangeTop;
		uniform float uRadius;

		const float PI = 3.141592657;
		
		float rand(vec2 co){
		    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
		}

		void main() {
		    vec2 uv = gl_FragCoord.xy / resolution.xy;
		    
			vec3 pos    = texture2D(texturePosition, uv).rgb;
			vec3 vel    = texture2D(textureVelocity, uv).rgb;

			pos += vel;
			//if(pos.x < .1) pos.x = 0.1;
			pos.x = max(pos.x, 1.0);
			if(pos.z > PI * 2.0) pos.z -= PI * 2.0;

			if(pos.y < -uRange) {
				pos.y = uRangeTop - 10.0;
				float randR = (rand(vec2(uTime))*.3) * .9;
				pos.x = randR * uRadius * 1.5;
			}

			gl_FragColor = vec4(pos, 1.0);
		}
	`;

	const texture_width = 512;
	const texture_height = 512	;

	const AMOUNT = texture_width * texture_height;
	const seed = Math.random() * 0xff;

	let uTime = Math.random() * 0xFF;
	let gpuCompute;

	let variables = {
    	TEXTURE_WIDTH:texture_width,
    	TEXTURE_HEIGHT:texture_height,
    	AMOUNT:AMOUNT
    }

    const computeVars = {

    }

	const rttTextures = {

	}

	let _radius = 400.0;
	let _range = 250.0;
	let _rangeTop = 250;

	let randomTexture;
	let posUniforms, velUniforms, particleUniforms;
	let maxSpeed = options.touch ? 60 / 90 : 1;
	let speedOffset = maxSpeed;
	let oldRtt;
	let material, mesh;
	let cnt = 0;

	let _shader, _shader2;

	const copyMaterial = new ShaderMaterial({
        uniforms: {
            resolution: { type: 'v2', value: new Vector2( texture_width, texture_height ) },
            uTexture: { type: 't', value: null }
        },
        vertexShader: `
        	varying vec2 vUv;

			void main() {
			    gl_Position = vec4( position, 1.0 );
			    vUv = uv;
			}
        `,
        fragmentShader: `
        	varying vec2 vUv;

			uniform vec2 resolution;
			uniform sampler2D uTexture;

			void main() {
			    vec2 uv = gl_FragCoord.xy / resolution.xy;
			    //vec2 uv = vUv;

			    vec3 color = texture2D( uTexture, uv ).xyz;
			    gl_FragColor = vec4( color, 1.0 );
			}
        `
    });

    const fboScene = new Scene();
    const fboCamera = new Camera();

    fboCamera.position.z = 1;

    const fboMesh = new Mesh( new PlaneBufferGeometry( 2, 2 ), copyMaterial );
    fboScene.add( fboMesh );

    const initOldRtt = () => {
		oldRtt = new WebGLRenderTarget( texture_width, texture_height, {
            wrapS: ClampToEdgeWrapping,
            wrapT: ClampToEdgeWrapping,
            minFilter: NearestFilter,
            magFilter: NearestFilter,
            format: RGBAFormat,
            type: FloatType,
            depthWrite: false,
            depthBuffer: false,
            stencilBuffer: false
        });
	};

	const copyTexture = (input, output) => {
        _fboMesh.material = copyMaterial;
        _copyShader.uniforms.uTexture.value = input.texture;
        
        renderer.setRenderTarget(output);
        renderer.render( _fboScene, _fboCamera );
        renderer.setRenderTarget(null);
    };

	const createTexture = () => {
        let texture = new DataTexture( new Float32Array( AMOUNT * 4 ), texture_width, texture_height, RGBAFormat, FloatType );
        texture.minFilter = NearestFilter;
        texture.magFilter = NearestFilter;
        texture.needsUpdate = true;
        texture.generateMipmaps = false;
        texture.flipY = false;
        
        return texture;
    };

	const initGPUCompute = _ => {
		console.log('application initGPUCompute')
		const rttSize = texture_width
		console.log(`rttSize:${rttSize}`)

		gpuCompute = new GPUComputationRenderer(rttSize, rttSize, renderer);

		rttTextures['positionTexture'] = gpuCompute.createTexture();
 		rttTextures['velocityTexture'] = gpuCompute.createTexture();

 		initPositionTexture(rttTextures['positionTexture'], 3)
 		initVelocityTexture(rttTextures['velocityTexture'])

 		randomTexture = createTexture();
 		initRandomTexture(randomTexture)

 		rttTextures['positionTexture'].needsUpdate = true;

 		const comPosition = computeVars['comPosition'] = gpuCompute.addVariable('texturePosition', computePosShader, rttTextures['positionTexture'])
    	const comVelocity = computeVars['comVelocity'] = gpuCompute.addVariable('textureVelocity', computeVelShader, rttTextures['velocityTexture'])

    	gpuCompute.setVariableDependencies(comPosition, [comPosition, comVelocity])

    	oldRtt = rttTextures['positionTexture'];
    	rttTextures['defaultPosTexture'] = rttTextures['positionTexture'].clone();

    	posUniforms = comPosition.material.uniforms
    	posUniforms.uTime = { type: 'f' , value:uTime }
    	posUniforms.uRange = { type:'f', value:_range}
    	posUniforms.uRangeTop = { type:'f', value:_rangeTop}
		posUniforms.uRadius = { type:'f', value:_radius}

    	gpuCompute.setVariableDependencies(comVelocity, [comVelocity, comPosition])
	    
	    velUniforms = comVelocity.material.uniforms
	    velUniforms.uTime = { type: 'f' , value:uTime }
	    velUniforms.textureRandom = { type: 't' , value: randomTexture	 }
	    velUniforms.defaultPosTexture = { type: 't', value: rttTextures['defaultPosTexture'].texture }
	    velUniforms.uHit =  { type:'v3', value:new Vector3() }
		velUniforms.uIsMouseDown =  { type:'f', value:0.0 }
		velUniforms.uNoiseScale = { type:'f', value:0.05}
		velUniforms.uRange = { type:'f', value:_range}
		velUniforms.uRangeTop = { type:'f', value:_rangeTop}
		velUniforms.uRadius = { type:'f', value:_radius}
		//velUniforms.uMinRadius = { type:'f', value:2.0}
		velUniforms.uMaxRadius = { type:'f', value:12.0}
		velUniforms.uSpeed = { type:'f', value:1.8 }
	    //initDepthRTT()
	    gpuCompute.init()
	};

	const initPositionTexture = (texture, mode = 0) => {
		const data = texture.image.data
		//console.log(mesh.geometry.attributes.position.count, variables.AMOUNT)
		const rand = MathUtils.randFloat;

		for(let i = 0; i < AMOUNT; i++){
			const radius = rand(8,8)
      		const phi = (Math.random() - 0.5) * Math.PI
      		const theta = Math.random() * Math.PI * 2

      		if(mode == 0){ // SPHERE
      			data[i * 4] = radius * Math.cos(theta) * Math.cos(phi)
		      	data[i * 4 + 1] = radius * Math.sin(phi)
		      	data[i * 4 + 2] = radius * Math.sin(theta) * Math.cos(phi)
		      	data[i * 4 + 3] = rand(0., 1.0)
      		}else if(mode == 1){  // CUBE   			
				data[i * 4] = rand(-5, 5)
				data[i * 4 + 1] = rand(-5, 5)
				data[i * 4 + 2] = rand(-5, 5)
				data[i * 4 + 3] = rand(0., 1.0)
      		}else if(mode == 2){ // PLANE
      			const DIM = 130

      			data[i * 4 + 0] = DIM / 2 - DIM * (i % texture_width) / texture_width
				data[i * 4 + 1] = DIM / 2 - DIM * ~~(i / texture_width) / texture_height
				data[i * 4 + 2] = 0	
				data[i * 4 + 3] = rand(0., 1.0)
      		
      		}else if(mode == 3){ // PLANE
      			data[i * 4] = rand(10, 200)
				data[i * 4 + 1] = rand(-_range, _range);//rand(-_range, _range)
				data[i * 4 + 2] = Math.random() * Math.PI * 2.0;
				data[i * 4 + 3] = 1.0
      		}else{
      			console.log(mesh.geometry.attributes.position.count)
      		}
		}

		texture.needsUpdate = true;
	};

	const initVelocityTexture = texture => {
		const data = texture.image.data;
		const rand = MathUtils.randFloat;

		for(let i=0; i<AMOUNT; i++){
			
			data[i * 4] = 0;//rand(-1.0, 1.0)
			data[i * 4 + 1] = 0;//rand(-1.0, 1.0)
			data[i * 4 + 2] = 0;//rand(-1.0, 1.0)
			data[i * 4 + 3] = 1.;//1.
		}
		
		texture.needsUpdate = true;
	};

	const initRandomTexture = texture => {
		const data = texture.image.data;
		const rand = MathUtils.randFloat;

		for(let i=0; i<AMOUNT; i++){
			data[i * 4] = rand(0, 1)
			data[i * 4 + 1] = rand(0, 1)
			data[i * 4 + 2] = rand(2, 8)
			data[i * 4 + 3] = Math.random()
		}

		texture.needsUpdate = true;
	};

	const initGeometry = () => {
		const rand = MathUtils.randFloat;
		const scale = 0.30;
		const size = 0.5;
		
		const baseGeometry = new TetrahedronBufferGeometry(.4);//( .5, .25, .025 );
		const geometry = new InstancedBufferGeometry();

		geometry.index = baseGeometry.index;
		geometry.attributes.position = baseGeometry.attributes.position;
		geometry.attributes.uv = baseGeometry.attributes.uv;
		geometry.attributes.normal = baseGeometry.attributes.normal;

		const offsets = new Float32Array(AMOUNT * 3);	
			
		const uv = new Float32Array(AMOUNT * 2);

		for(let i = 0; i < AMOUNT; i++) {
			const index = i * 2

	      	uv[index + 0] = (i % texture_width) / texture_width
	      	uv[index + 1] = ~~(i / texture_width) / texture_height

	      	/*const indexPos = i * 3;
      	
      		offsets[indexPos + 0] = rand(-10, 10);//(i % texture_width) / texture_width;
    		offsets[indexPos + 1] = rand(-10, 10);//0;
			offsets[indexPos + 2] = rand(-10, 10);//~~(i / texture_width) / texture_height;*/
		}

		//geometry.setAttribute('offset', new InstancedBufferAttribute(offsets, 3).setUsage(DynamicDrawUsage))
		geometry.setAttribute('uv2', new InstancedBufferAttribute(uv, 2))

		material = new MeshStandardMaterial({
			color:new Color(0xFFFFFF),
			transparent:true,
			/*metalness:.1,
			roughness:.8,*/
			blending:AdditiveBlending
		});	

		material.onBeforeCompile = (shader) => {
			shader.uniforms.uTexturePos =  { type:'t', value:null }
			shader.uniforms.uTextureOldPos =  { type:'t', value:null }
			shader.uniforms.uTextureRandom =  { type:'t', value:null }
			shader.uniforms.uTime =  { type:'f', value:uTime }
			shader.uniforms.uColor = { type:'t', value:new TextureLoader().load('./textures/blue.jpg')}
			_shader = shader;

			shader.vertexShader = shader.vertexShader.replace(
				'#include <common>',
				`
				#include <common>

				//attribute vec3 offset;
				attribute vec2 uv2;

				varying vec3 vColor;
				varying vec2 vTextureCoord;
				varying float vOpacity;


				uniform sampler2D uTexturePos;
				uniform sampler2D uTextureOldPos;

				uniform sampler2D uTextureRandom;
				uniform float uTime;

				const vec3 FRONT = vec3(0.0, 0.0, -1.0);
				const vec3 UP = vec3(0.0, 1.0, 0.0);

				mat4 rotationMatrix(vec3 axis, float angle) {
				    axis = normalize(axis);
				    float s = sin(angle);
				    float c = cos(angle);
				    float oc = 1.0 - c;
				    
				    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
				                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
				                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
				                0.0,                                0.0,                                0.0,                                1.0);
				}

				vec3 rotate(vec3 v, vec3 axis, float angle) {
					mat4 m = rotationMatrix(axis, angle);
					return (m * vec4(v, 1.0)).xyz;
				}

				vec3 getPos(vec3 value) {
					vec3 pos;

					pos.y = value.y;
					pos.x = cos(value.z) * value.x;
					pos.z = sin(value.z) * value.x;
					return pos;
				}
				`
			);

			shader.vertexShader = shader.vertexShader.replace(
				'#include <begin_vertex>',
				`
				//#include <begin_vertex>

				/*vec3 oldPos  = texture2D(uTextureOldPos, uv2).rgb;
				vec3 pos = texture2D(uTexturePos, uv2).rgb;
				vec3 extra = texture2D(uTextureRandom, uv2).rgb;

				vec3 finalPos      = pos;//mix(oldPos, pos, .5);

				vec3 dir      = normalize(oldPos - finalPos);
				vec3 axis     = cross(dir, FRONT);
				float theta   = acos(dot(dir, FRONT));

				float scale   = sin(extra.z + uTime * extra.r) * .4 + .6;
				
				vec3 p = position;
				p *= scale;
				
				p 	  = rotate(p.xyz, axis, theta) + finalPos;
				vec3 transformed = vec3(p);

				vNormal = rotate(normal, axis, theta);

				float c       = mix(extra.r, 1.0, .8);
				vColor        = vec3(c);*/
				vec3 extra = texture2D(uTextureRandom, uv2).rgb;

				//	vec3 pos = getPos(position);
				//vec2 uv = uv * 25.;
				vec3 pos = texture2D(uTexturePos, uv2).rgb;
				pos = getPos(pos);
				pos.y -= 25.;
			    
			    float scale = extra.b;

			    vec3 transformed = vec3(position * scale + pos);
			    vTextureCoord = extra.xy;

			    float c = sin(uTime * mix(extra.x, 1.0, .5));
    			vOpacity = smoothstep(.5, 1.0, c);
				`
			);

			shader.fragmentShader = shader.fragmentShader.replace(
				'#include <common>',
				`
				#include <common>
				
				uniform sampler2D uColor;
				varying vec3 vColor;
				varying vec2 vTextureCoord;
				varying float vOpacity;

				const vec2 center = vec2(.5);

				float diff(vec3 N, vec3 L) {
					return max(dot(N, normalize(L)), 0.0);
				}


				vec3 diff(vec3 N, vec3 L, vec3 C) {
					return diff(N, L) * C;
				}
				`
			);

			shader.fragmentShader = shader.fragmentShader.replace(
				'#include <map_fragment>',
				`#include <map_fragment>

				if(distance(center, gl_PointCoord) > .8) discard;


				float d = diff(vNormal, vec3(0, 0, 0));

				d = mix(d, 1.0, .5);
				//diffuseColor.a = vOpacity;
				vec3 color = texture2D(uColor, vTextureCoord).rgb;

				diffuseColor = vec4(vec3(d) * color,vOpacity);
				`
			);
		};

		mesh = new Mesh(geometry, material);

			/*mesh.customDepthMaterial = new MeshDepthMaterial({
		      depthPacking: RGBADepthPacking,
		      alphaTest: 0.5
		    });*/

	    /*mesh.customDepthMaterial.onBeforeCompile = (shader) => {
			shader.uniforms.uTexturePos =  { type:'t', value:null }
			shader.uniforms.uTextureOldPos =  { type:'t', value:null }
			shader.uniforms.uTextureRandom =  { type:'t', value:null }
			shader.uniforms.uTime =  { type:'f', value:uTime }

			_shader2 = shader;

			shader.vertexShader = shader.vertexShader.replace(
				'#include <common>',
				`
				#include <common>
				varying vec3 vNormal;

				#define DEPTH_PACKING 3201
				
				//attribute vec3 offset;
				attribute vec2 uv2;

				varying vec3 vColor;

				uniform sampler2D uTexturePos;
				uniform sampler2D uTextureOldPos;

				uniform sampler2D uTextureRandom;
				uniform float uTime;

				const vec3 FRONT = vec3(0.0, 0.0, -1.0);
				const vec3 UP = vec3(0.0, 1.0, 0.0);

				mat4 rotationMatrix(vec3 axis, float angle) {
				    axis = normalize(axis);
				    float s = sin(angle);
				    float c = cos(angle);
				    float oc = 1.0 - c;
				    
				    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
				                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
				                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
				                0.0,                                0.0,                                0.0,                                1.0);
				}

				vec3 rotate(vec3 v, vec3 axis, float angle) {
					mat4 m = rotationMatrix(axis, angle);
					return (m * vec4(v, 1.0)).xyz;
				}
				`
			);

			shader.vertexShader = shader.vertexShader.replace(
				'#include <begin_vertex>',
				`
				//#include <begin_vertex>

				vec3 oldPos  = texture2D(uTextureOldPos, uv2).rgb;
				vec3 pos = texture2D(uTexturePos, uv2).rgb;
				vec3 extra = texture2D(uTextureRandom, uv2).rgb;

				vec3 finalPos      = pos;//mix(oldPos, pos, .5);

				vec3 dir      = normalize(oldPos - finalPos);
				vec3 axis     = cross(dir, FRONT);
				float theta   = acos(dot(dir, FRONT));

				float scale   = sin(extra.z + uTime * extra.r) * .4 + .6;
				
				vec3 p = position;
				p *= scale;
				
				p 	  = rotate(p.xyz, axis, theta) + finalPos;
				vec3 transformed = vec3(p);

				vNormal = rotate(normal, axis, theta);

				float c       = mix(extra.r, 1.0, .8);
				vColor        = vec3(c);

				`
			);

			shader.fragmentShader = '#define DEPTH_PACKING 3201\n' + shader.fragmentShader;

			shader.fragmentShader = shader.fragmentShader.replace(
				'#include <common>',
				`
				#include <common>

				varying vec3 vNormal;
				varying vec3 vColor;

				`
			);
		};*/

		mesh.castShadow = true;
		mesh.receiveShadow = true;

		scene.add(mesh);

		mesh.frustumCulled = false;

		renderer.compile(scene, camera);
	};

	const update = (delta, uTime, mouse3d, isMouseDown) => {
		gpuCompute.compute();

		let f = isMouseDown ? 10.0 : 0.0;
		const r = isMouseDown ? 8.0 : 2.0;

		posUniforms.uTime.value += .01;
		velUniforms.uTime.value += .01;

		velUniforms.uHit.value.copy(mouse3d);
		velUniforms.uIsMouseDown.value = f;
		//console.log(velUniforms.uTime.value);

		const comPosition = computeVars['comPosition']
		const comVelocity = computeVars['comVelocity'] 

		//dirLight2.position.copy( mouse3d );

		if(_shader){

		
			//console.log(_shader.uniforms);
			_shader.uniforms.uTexturePos.value = gpuCompute.getCurrentRenderTarget(comPosition).texture;
			_shader.uniforms.uTextureRandom.value = randomTexture;
			
			/*if(oldRtt){
				if(cnt % 5 == 0){
					//copyTexture(gpuCompute.getCurrentRenderTarget(comPosition),oldRtt);
					oldRtt = gpuCompute.getCurrentRenderTarget(comPosition).clone();

					_shader.uniforms.uTextureOldPos.value = oldRtt.texture;
				}
				
			}*/

			_shader.uniforms.uTime.value += .01;

			//_shader.uniforms.uTextureVel.value = gpuCompute.getCurrentRenderTarget(comVelocity).texture ;
		}

		/*if(_shader2){

		
			//console.log(_shader.uniforms);
			_shader2.uniforms.uTexturePos.value = gpuCompute.getCurrentRenderTarget(comPosition).texture;
			_shader2.uniforms.uTextureRandom.value = randomTexture;
			
			if(oldRtt){
				if(cnt % 5 == 0){
					//copyTexture(gpuCompute.getCurrentRenderTarget(comPosition),oldRtt);
					oldRtt = gpuCompute.getCurrentRenderTarget(comPosition).clone();

					_shader2.uniforms.uTextureOldPos.value = oldRtt.texture;
				}
				
			}

			_shader2.uniforms.uTime.value += .01;

			//_shader.uniforms.uTextureVel.value = gpuCompute.getCurrentRenderTarget(comVelocity).texture ;
		}*/

		//oldRtt = gpuCompute.getCurrentRenderTarget(comPosition).clone();
		//console.log(oldRtt)
	};

	
	initGeometry();
	initGPUCompute();

	const base = {
		update,
	};

	Object.defineProperty(base, 'range', {
		get:() => _range,
		set:(value) => {
			_range = value;
			velUniforms.uRange.value = _range;
			posUniforms.uRange.value = _range;
		},
	});

	Object.defineProperty(base, 'radius', {
		get:() => _radius,
		set:(value) => {
			_radius = value;
			velUniforms.uRadius.value = _radius;
			posUniforms.uRange.value = _radius;
		},
	});
	
	return base;
};

export { Flocking };
