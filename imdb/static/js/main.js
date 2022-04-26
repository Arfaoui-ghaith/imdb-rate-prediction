/**
* Template Name: Anyar - v4.7.1
* Template URL: https://bootstrapmade.com/anyar-free-multipurpose-one-page-bootstrap-theme/
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/
(async function() {
  "use strict";

  let genreList = await axios.get('http://localhost:5000/genrelist')
  console.log(genreList.data)

  let options = `<option value='${Object.values(genreList.data)[0]*1}' >${Object.keys(genreList.data)[0]}</option>`
  for(let i=1; i<Object.keys(genreList.data).length;i++){
    options = `<option value='${Object.values(genreList.data)[i]*1}' >${Object.keys(genreList.data)[i]}</option>`
  }

  let div = document.getElementById("predict");
  let tree = document.getElementById("tree");
  let knn = document.getElementById("knn");
  let bayes = document.getElementById("bayes");
  let means = document.getElementById("means");
  
  
  let ch = `  <section class="contact" >
  <div class="container" data-aos="fade-up" >

    <div class="section-title">
      <h2>%title%</h2>
    </div>

    <div class="row mt-1 d-flex justify-content-center" data-aos="fade-right" data-aos-delay="100">
      <div class="col-lg-6 mt-5 mt-lg-0" data-aos="fade-left" data-aos-delay="100">
        <div class="php-email-form">
          
          <div class="form-group mt-3">
            <input id="title" type="text" name="Title" class="form-control" id="name" placeholder="Movie Title" required>
          </div>

          <div class="form-group mt-3">
            <input id="year" type="number" class="form-control" name="email" id="email" placeholder="Year"  required>
          </div>

          <div class="form-group mt-3">
            <input id="duration" type="number" class="form-control" name="email" id="email" placeholder="Duration in minutes" step="0.01" required>
          </div>
         
          <div class="form-group mt-3">
            <select id="genre" class="form-control" name="subject" id="subject" required>
              <option value='${Object.values(genreList.data)[1]*1}' selected>${Object.keys(genreList.data)[1]}</option>
              <option value='${Object.values(genreList.data)[2]*1}' >${Object.keys(genreList.data)[2]}</option>
              <option value='${Object.values(genreList.data)[3]*1}' >${Object.keys(genreList.data)[3]}</option>
              <option value='${Object.values(genreList.data)[4]*1}' >${Object.keys(genreList.data)[4]}</option>
              <option value='${Object.values(genreList.data)[5]*1}' >${Object.keys(genreList.data)[5]}</option>
              <option value='${Object.values(genreList.data)[6]*1}' >${Object.keys(genreList.data)[6]}</option>
              <option value='${Object.values(genreList.data)[7]*1}' >${Object.keys(genreList.data)[7]}</option>
              <option value='${Object.values(genreList.data)[8]*1}' >${Object.keys(genreList.data)[8]}</option>
              <option value='${Object.values(genreList.data)[9]*1}' >${Object.keys(genreList.data)[9]}</option>
              <option value='${Object.values(genreList.data)[10]*1}' >${Object.keys(genreList.data)[10]}</option>
              <option value='${Object.values(genreList.data)[11]*1}' >${Object.keys(genreList.data)[11]}</option>
              <option value='${Object.values(genreList.data)[12]*1}' >${Object.keys(genreList.data)[12]}</option>
              <option value='${Object.values(genreList.data)[13]*1}' >${Object.keys(genreList.data)[13]}</option>
              <option value='${Object.values(genreList.data)[14]*1}' >${Object.keys(genreList.data)[14]}</option>
              <option value='${Object.values(genreList.data)[15]*1}' >${Object.keys(genreList.data)[15]}</option>
              <option value='${Object.values(genreList.data)[16]*1}' >${Object.keys(genreList.data)[16]}</option>
              <option value='${Object.values(genreList.data)[17]*1}' >${Object.keys(genreList.data)[17]}</option>
              <option value='${Object.values(genreList.data)[18]*1}' >${Object.keys(genreList.data)[18]}</option>
              <option value='${Object.values(genreList.data)[19]*1}' >${Object.keys(genreList.data)[19]}</option>
              <option value='${Object.values(genreList.data)[20]*1}' >${Object.keys(genreList.data)[20]}</option>
            </select>
          </div>
          <div class="form-group mt-3">
            <input type="number" id="votes" class="form-control" name="message" rows="5" placeholder="Votes" step="0.01" required></input>
          </div>
          <div class="my-3">
            <div class="loading">Loading</div>
            <div class="error-message"></div>
            <div class="sent-message">Your message has been sent. Thank you!</div>
          </div>
          <div class="text-center"><button id="sendRequest" type="submit">Send Message</button></div>
          <div id="res" class="text-center">
            
          </div>
        </div>

      </div>

    </div>

  </div>
</section><!-- End Contact Section -->`

  let algorithm;

  tree.addEventListener('click', (e) => {
    div.innerHTML = ch.replace('%title%','Tree Decision');
    algorithm = 'Tree'

    let button = document.getElementById("sendRequest");

    button.addEventListener('click', async (e) => {
      let title = document.getElementById("title").value
      let year = document.getElementById("year").value
      let duration = document.getElementById("duration").value
      let genre = document.getElementById("genre").value
      let votes = document.getElementById("votes").value
  
      if(year.length > 0 && duration.length > 0 && genre.length > 0 && votes.length > 0 && title.length > 0){
        let res = await axios.post(`http://localhost:5000/${algorithm}/${year}/${duration}/${genre}/${votes}`)
        console.log(res.data)
        document.getElementById("res").innerHTML = `<div class="alert alert-success mt-4" role="alert">
        ${title} ${res.data}
      </div>`
      }
      
    })

  })

  knn.addEventListener('click', (e) => {
    div.innerHTML = ch.replace('%title%','K-Nearest Neighbors');
    algorithm = 'KNN'

    let button = document.getElementById("sendRequest");

    button.addEventListener('click', async (e) => {
      let title = document.getElementById("title").value
      let year = document.getElementById("year").value
      let duration = document.getElementById("duration").value
      let genre = document.getElementById("genre").value
      let votes = document.getElementById("votes").value
  
      if(year.length > 0 && duration.length > 0 && genre.length > 0 && votes.length > 0 && title.length > 0){
        let res = await axios.post(`http://localhost:5000/${algorithm}/${year}/${duration}/${genre}/${votes}`)
        console.log(res.data)
        document.getElementById("res").innerHTML = `<div class="alert alert-success mt-4" role="alert">
        ${title} ${res.data}
      </div>`
      }
      
    })

  })

  bayes.addEventListener('click', (e) => {
    div.innerHTML = ch.replace('%title%','Naive Bayes');
    algorithm = 'Naive_bayes'

    let button = document.getElementById("sendRequest");

    button.addEventListener('click', async (e) => {
      let title = document.getElementById("title").value
      let year = document.getElementById("year").value
      let duration = document.getElementById("duration").value
      let genre = document.getElementById("genre").value
      let votes = document.getElementById("votes").value
  
      if(year.length > 0 && duration.length > 0 && genre.length > 0 && votes.length > 0 && title.length > 0){
        let res = await axios.post(`http://localhost:5000/${algorithm}/${year}/${duration}/${genre}/${votes}`)
        console.log(res.data)
        document.getElementById("res").innerHTML = `<div class="alert alert-success mt-4" role="alert">
        ${title} ${res.data}
      </div>`
      }
      
    })

  })

  means.addEventListener('click', (e) => {
    div.innerHTML = ch.replace('%title%','K-Means');
  })

  
  




  /**
   * Easy selector helper function
   */
  const select = (el, all = false) => {
    el = el.trim()
    if (all) {
      return [...document.querySelectorAll(el)]
    } else {
      return document.querySelector(el)
    }
  }

  /**
   * Easy event listener function
   */
  const on = (type, el, listener, all = false) => {
    let selectEl = select(el, all)
    if (selectEl) {
      if (all) {
        selectEl.forEach(e => e.addEventListener(type, listener))
      } else {
        selectEl.addEventListener(type, listener)
      }
    }
  }

  /**
   * Easy on scroll event listener 
   */
  const onscroll = (el, listener) => {
    el.addEventListener('scroll', listener)
  }

  /**
   * Navbar links active state on scroll
   */
  let navbarlinks = select('#navbar .scrollto', true)
  const navbarlinksActive = () => {
    let position = window.scrollY + 200
    navbarlinks.forEach(navbarlink => {
      if (!navbarlink.hash) return
      let section = select(navbarlink.hash)
      if (!section) return
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        navbarlink.classList.add('active')
      } else {
        navbarlink.classList.remove('active')
      }
    })
  }
  window.addEventListener('load', navbarlinksActive)
  onscroll(document, navbarlinksActive)

  /**
   * Scrolls to an element with header offset
   */
  const scrollto = (el) => {
    let header = select('#header')
    let offset = header.offsetHeight

    if (!header.classList.contains('fixed-top')) {
      offset += 70
    }

    let elementPos = select(el).offsetTop
    window.scrollTo({
      top: elementPos - offset,
      behavior: 'smooth'
    })
  }

  /**
   * Header fixed top on scroll
   */
  let selectHeader = select('#header')
  let selectTopbar = select('#topbar')
  if (selectHeader) {
    const headerScrolled = () => {
      if (window.scrollY > 100) {
        selectHeader.classList.add('header-scrolled')
        if (selectTopbar) {
          selectTopbar.classList.add('topbar-scrolled')
        }
      } else {
        selectHeader.classList.remove('header-scrolled')
        if (selectTopbar) {
          selectTopbar.classList.remove('topbar-scrolled')
        }
      }
    }
    window.addEventListener('load', headerScrolled)
    onscroll(document, headerScrolled)
  }

  /**
   * Back to top button
   */
  let backtotop = select('.back-to-top')
  if (backtotop) {
    const toggleBacktotop = () => {
      if (window.scrollY > 100) {
        backtotop.classList.add('active')
      } else {
        backtotop.classList.remove('active')
      }
    }
    window.addEventListener('load', toggleBacktotop)
    onscroll(document, toggleBacktotop)
  }

  /**
   * Mobile nav toggle
   */
  on('click', '.mobile-nav-toggle', function(e) {
    select('#navbar').classList.toggle('navbar-mobile')
    this.classList.toggle('bi-list')
    this.classList.toggle('bi-x')
  })

  /**
   * Mobile nav dropdowns activate
   */
  on('click', '.navbar .dropdown > a', function(e) {
    if (select('#navbar').classList.contains('navbar-mobile')) {
      e.preventDefault()
      this.nextElementSibling.classList.toggle('dropdown-active')
    }
  }, true)

  /**
   * Scrool with ofset on links with a class name .scrollto
   */
  on('click', '.scrollto', function(e) {
    if (select(this.hash)) {
      e.preventDefault()

      let navbar = select('#navbar')
      if (navbar.classList.contains('navbar-mobile')) {
        navbar.classList.remove('navbar-mobile')
        let navbarToggle = select('.mobile-nav-toggle')
        navbarToggle.classList.toggle('bi-list')
        navbarToggle.classList.toggle('bi-x')
      }
      scrollto(this.hash)
    }
  }, true)

  /**
   * Scroll with ofset on page load with hash links in the url
   */
  window.addEventListener('load', () => {
    if (window.location.hash) {
      if (select(window.location.hash)) {
        scrollto(window.location.hash)
      }
    }
  });

  /**
   * Preloader
   */
  let preloader = select('#preloader');
  if (preloader) {
    window.addEventListener('load', () => {
      preloader.remove()
    });
  }

  /**
   * Clients Slider
   */
  new Swiper('.clients-slider', {
    speed: 400,
    loop: true,
    autoplay: {
      delay: 5000,
      disableOnInteraction: false
    },
    slidesPerView: 'auto',
    pagination: {
      el: '.swiper-pagination',
      type: 'bullets',
      clickable: true
    },
    breakpoints: {
      320: {
        slidesPerView: 2,
        spaceBetween: 40
      },
      480: {
        slidesPerView: 3,
        spaceBetween: 60
      },
      640: {
        slidesPerView: 4,
        spaceBetween: 80
      },
      992: {
        slidesPerView: 6,
        spaceBetween: 120
      }
    }
  });

  /**
   * Porfolio isotope and filter
   */
  window.addEventListener('load', () => {
    let portfolioContainer = select('.portfolio-container');
    if (portfolioContainer) {
      let portfolioIsotope = new Isotope(portfolioContainer, {
        itemSelector: '.portfolio-item',
        layoutMode: 'fitRows'
      });

      let portfolioFilters = select('#portfolio-flters li', true);

      on('click', '#portfolio-flters li', function(e) {
        e.preventDefault();
        portfolioFilters.forEach(function(el) {
          el.classList.remove('filter-active');
        });
        this.classList.add('filter-active');

        portfolioIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        portfolioIsotope.on('arrangeComplete', function() {
          AOS.refresh()
        });
      }, true);
    }

  });

  /**
   * Initiate portfolio lightbox 
   */
  const portfolioLightbox = GLightbox({
    selector: '.portfolio-lightbox'
  });

  /**
   * Initiate glightbox 
   */
  const gLightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Portfolio details slider
   */
  new Swiper('.portfolio-details-slider', {
    speed: 400,
    loop: true,
    autoplay: {
      delay: 5000,
      disableOnInteraction: false
    },
    pagination: {
      el: '.swiper-pagination',
      type: 'bullets',
      clickable: true
    }
  });

  /**
   * Animation on scroll
   */
  window.addEventListener('load', () => {
    AOS.init({
      duration: 1000,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    })
  });

})()