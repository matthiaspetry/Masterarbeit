const NavBar = () => {
  return (
    <nav className="bg-blue-600 text-white px-4 py-3">
      <div className="container mx-auto flex justify-between items-center">
        <div className="font-semibold text-xl">BrandName</div>
        <ul className="flex space-x-4">
          <li>
           
              <a className="hover:text-blue-300">Home</a>
            
          </li>
          <li>
            
              <a className="hover:text-blue-300">About</a>
          
          </li>
          <li>
            
              <a className="hover:text-blue-300">Contact</a>
           
          </li>
          {/* Additional navigation links */}
        </ul>
      </div>
    </nav>
  );
};

export default NavBar;