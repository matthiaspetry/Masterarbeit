const NavBar = () => {
  return (
    <nav className="bg-blue-700 text-white px-4 py-3 ">
      <div className="container flex justify-between items-center">
        <div className="flex items-center transition duration-300 ease-in-out p-2 rounded">
          <img src="/mechanical-arm-icon.png" alt="RoboRetiever Logo" className="h-10 mr-3" />
          <span className="font-bold text-2xl tracking-wide">RoboRetriever</span>
        </div>
        {/* Add other nav items here if needed */}
      </div>
    </nav>
  );
};

export default NavBar;