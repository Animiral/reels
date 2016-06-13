/**
 * This is the C++ implementation of the distance-to-goal heuristic component of the reels.py application.
 * The file README contains a general description of the program. Read on for implementation details.
 *
 * The heuristic function (estimated distance to goal) for a partial reel solution is calculated using its own tree search algorithm.
 * The estimate search attempts to fit the remaining free pieces plus the start and end pieces of the partial solution sequence
 * together in left-right pairs such that no piece is assigned twice and the number of overlapping symbols between these remaining
 * pieces is maximized.
 *
 * This method improves on a more basic heuristic in which conflicts in left-right assignments are ignored and all pieces are simply
 * assumed to match up with their preferred counterpart. The basic heuristic is faster, but also produces more nodes for the reel
 * search to go through until it finds the optimal solution.
 *
 * With every step deeper into the estimate search tree, the algorithm resolves one more conflict between two pieces tied for the
 * same right-piece by re-assigning it to a different right-piece. This results in a breadth-first search with a constant branching
 * factor of 2.
 *
 * Since the algorithm produces a continuous improvement over the basic heuristic, it can be interrupted at any time. It can e.g. be
 * limited to a maximum number of examined or memorized nodes.
 *
 * This C++ solution is implemented without such limits. Its primary purpose is to run the complete heuristic and be fast about it.
 */

#include <vector>
#include <queue>
#include <memory>
#include <algorithm>

// Temp debug stuff
#include <iostream>
#include <string>
#include <sstream>
class EstContext;
class ReelContext;
__attribute__((unused)) static std::string debug_assoc_str(const std::vector<int>& assoc);
__attribute__((unused)) static std::string debug_est_context_str(const EstContext& context);
__attribute__((unused)) static std::string debug_reel_context_str(const ReelContext& context);
#ifndef LOG_ENABLED
#define LOG_ENABLED false
#endif
// End Temp debug stuff

using std::size_t;

using piece_t = int; // piece_t is actually an index into numerous arrays, such as overmat, pref, lobs, ...
class ReelContext;

extern "C"
{
	ReelContext* create_context(size_t n, int* overmat, size_t p, piece_t* pref, int* lobs) noexcept;
	void destroy_context(const ReelContext* context) noexcept;
	int estimate(size_t l, piece_t lefts[], piece_t a, piece_t z, const ReelContext* context) noexcept;
}

const piece_t NONE = -1; // sentinel marker in lists such as pref

template<bool enable = true>
class LogImpl {
public:
	operator bool() { return bool(std::cerr); }

	template<typename T> LogImpl& operator<<(const T& t)
	{
		std::cerr << t;
		return *this;
	}
};

template<>
class LogImpl<false>
{
public:
	operator bool() { return false; }
	template<typename T> LogImpl& operator<<(const T& t) { return *this; }
};

using Log = LogImpl<LOG_ENABLED>;

/**
 * ReelContext is an object that holds precomputed global information relevant to the heuristic.
 * It is constructed once per application run around the input reel problem.
 * Then it informs the heuristic function in every reel search node.
 * The heuristic function only needs a const view on the context.
 */
class ReelContext
{

public:

	/**
	 * Construct the context from the free-form data as passed through the C interface.
	 * The data are copied into properly managed C++ containers and made accessible
	 * through this context from then on.
	 */
	ReelContext(size_t n, int overmat[], size_t p, piece_t pref[], int lobs[])
	: m_n(n), m_overmat(overmat, overmat+(n*n)), m_p(p), m_lobs(lobs, lobs+n)
	{
		for(size_t i = 0; i < n; i++) {
			piece_t* begin = pref + i*p;   // i-th row of pref
			piece_t* end = pref + (i+1)*p; // each row contains (n-1) pieces
			m_pref.emplace_back(begin, end);
		}
	}

	/**
	 * Return the number of obs pieces in the original input. This is also the extent (dimension)
	 * of the overmat, pref list-of-lists and lobs.
	 */
	size_t n() const
	{
		return m_n;
	}

	/**
	 * Return the overlap matrix entry corresponding to piece i as the left-hand and piece j as
	 * the right-hand pieces.
	 * There is no bounds checking in this getter, though the internal container checks the bounds
	 * to keep array access within the matrix. It is possible to overshoot a row.
	 */
	int overlap(piece_t i, piece_t j) const
	{
		return m_overmat[i*m_n + j];
	}

	/**
	 * Return the number of pieces in a single pref list.
	 */
	size_t p() const
	{
		return m_p;
	}

	/**
	 * Return the nth-ranked preferred overlap piece for the given left piece from the pref list.
	 */
	piece_t pref(piece_t left, size_t rank) const
	{
		return m_pref[left][rank];
	}

	/**
	 * Return the length of the given piece in symbols.
	 */
	int lobs(piece_t i) const
	{
		return m_lobs[i];
	}

	Log& log() const
	{
		return m_log;
	}

private:
	size_t m_n;
	std::vector<int> m_overmat;
	size_t m_p;
	std::vector< std::vector<piece_t> > m_pref;
	std::vector<int> m_lobs;
	mutable Log m_log;	

	friend std::string debug_reel_context_str(const ReelContext& );

};

/**
 * EstContext is a storage of data that supports the flyweight pattern implementation of EstNode.
 * It describes the current node of the reel search that must be estimated in this sub-search.
 */
class EstContext
{

public:
	EstContext(size_t l, piece_t lefts[], piece_t a, piece_t z)
	: m_lefts(lefts, lefts+l), m_a(a), m_z(z)
	{}

	std::vector<piece_t>& lefts()
	{
		return m_lefts;
	}

	const std::vector<piece_t>& lefts() const
	{
		return m_lefts;
	}

	const piece_t a() const
	{
		return m_a;
	}

	const piece_t z() const
	{
		return m_z;
	}

private:
	std::vector<piece_t> m_lefts;
	piece_t m_a;
	piece_t m_z;

};

class EstNode;

/**
 * The EstNode is one node in the tree search for a conflict-free association mapping between
 * left-hand and right-hand pieces.
 * One EstNode proposes one such mapping, which may contain association conflicts.
 * To travel from one EstNode to the next means to (try and) resolve one such conflict by
 * re-associating a conflicted left-piece to another right-piece.
 *
 * The EstNode is owned by the search algorithm using a unique_ptr, until the search examines
 * and discards it. At that point, ownership passes to the node’s children, who share it. This
 * saves space because a child does not have to remember its full assoc list until it expand()s.
 */
class EstNode
{

public:
	using Ptr = std::unique_ptr<EstNode>; // pointer type used for node successors

	/**
	 * Construct the root EstNode for a new search.
	 */
	EstNode()
	: m_parent(nullptr), m_resolution(NONE), m_cost(0)
	{}

	/**
	 * Construct an EstNode as child of another node.
	 * The child must later receive ownership over the parent, so this constructor only
	 * makes sense to call from within EstNode::successors.
	 */
	EstNode(piece_t resolution, int step, int cost)
	: m_resolution(resolution), m_step(step), m_cost(cost)
	{}

	/**
	 * Collect information about this node, including compilation of its assoc, goal test and successors.
	 * Requires a non-const context to reuse the space in est_context.lefts for sorting.
	 */
	bool expand(EstContext& est_context, const ReelContext& reel_context)
	{
		m_assoc = this->make_assoc(est_context, reel_context);

		piece_t p, q;
		bool conflict = this->find_conflict(est_context, reel_context, &p, &q);

		if(conflict) {
			EstNode::Ptr next = this->resolve(p, est_context, reel_context);
			if(next)
				m_succ.push_back(std::move(next));

			next = this->resolve(q, est_context, reel_context);
			if(next)
				m_succ.push_back(std::move(next));

			return true;
		}
		else {
			return false;
		}
	}

	/**
	 * Return the given node’s successors.
	 * Ownership of the node object transfers from the given Ptr to the returned children.
	 * The parent node will live as long as not all of its children have expand()ed.
	 */
	static std::vector<Ptr>& successors(Ptr self)
	{
		std::shared_ptr<EstNode> shared_self = std::move(self);

		for(auto& node : shared_self->m_succ) {
			node->m_parent = shared_self;
		}

		return shared_self->m_succ;
	}

	/*
	 * Getter function for the substance of the node.
	 * expand() must be called on this node before its assoc can be read.
	 */
	const std::vector<int>& assoc() const
	{
		return m_assoc;
	}

	/**
	 * Return this node’s cost value.
	 */
	int cost() const
	{
		return m_cost;
	}

private:
	std::shared_ptr<EstNode> m_parent;
	piece_t m_resolution;
	int m_step;
	int m_cost;
	std::vector<int> m_assoc;
	std::vector<Ptr> m_succ;

	friend class greater_cost;


	/**
	 * Build a new assoc vector from the parent’s assoc and this node’s own resolution piece.
	 */
	std::vector<int> make_assoc(EstContext& est_context, const ReelContext& reel_context)
	{
		if(m_parent) {
			std::vector<int> assoc = m_parent->m_assoc;
			assoc[m_resolution] += m_step;
			return assoc;
		}
		else {
			std::vector<int> assoc(reel_context.n(), -1);

			for(piece_t l : est_context.lefts()) {
				assoc[l] += this->step(assoc, l, est_context, reel_context);	
			}

			return assoc;
		}
	}

	/**
	 * Determine two pieces in this node which are associated with the same right-piece.
	 * The two pieces are written into the variables pointed to by p and q.
	 * Return true if such a conflict was found, false otherwise.
	 */
	bool find_conflict(EstContext& est_context, const ReelContext& reel_context, piece_t* p, piece_t* q)
	{
		std::vector<piece_t>& lefts = est_context.lefts();

		if(lefts.size() < 2) return false; // trivial case

		// Mapping function from left-piece to right-piece according to the current assoc
		auto right_of = [&assoc = m_assoc, &reel_context] (piece_t left) { return reel_context.pref(left, assoc[left]); };

		// Predicate to sort a list of left-pieces by their associated right-piece
		auto by_right = [right_of] (piece_t a, piece_t b) { return right_of(a) < right_of(b); };

		std::sort(lefts.begin(), lefts.end(), by_right);

		// If there are duplicates, they are now adjacent => do linear search
		piece_t prev = lefts[0];
		for(auto next = lefts.begin()+1; next != lefts.end(); ++next) {
			if(right_of(prev) == right_of(*next)) {
				*p = prev;
				*q = *next;
				return true;
			}
			prev = *next;
		}

		return false;
	}

	/**
	 * Construct a new EstNode as a child to this one in which the conflict involving
	 * the piece p is resolved by re-associating p with another right-piece.
	 * Return the new node in the appropriate Ptr.
	 * Returns a nullptr if the conflict cannot be resolved by re-associating p.
	 */
	EstNode::Ptr resolve(piece_t piece, const EstContext& est_context, const ReelContext& reel_context)
	{
		int step = this->step(m_assoc, piece, est_context, reel_context);

		if(step >= 0) {
			int rank = m_assoc[piece];
			piece_t rhs_before = reel_context.pref(piece, rank);
			piece_t rhs_after = reel_context.pref(piece, rank+step);
			int cost = m_cost + reel_context.overlap(piece, rhs_before) - reel_context.overlap(piece, rhs_after);

			return std::make_unique<EstNode> (piece, step, cost);
		}
		else {
			return nullptr;
		}
	}

	/**
	 * Return the number of ranks that the piece would have to advance in its pref list
	 * to find a new valid right-piece given the left/free lists in the EstContext.
	 * If the piece has already exhausted all pref ranks, returns -1.
	 */
	int step(const std::vector<int>& assoc, piece_t piece, const EstContext& est_context, const ReelContext& reel_context)
	{
		auto is_right = [&est_context] (piece_t right)
		{
			auto L = est_context.lefts();
			return (right == est_context.a()) || (std::find(L.begin(), L.end(), right) != L.end());
		};

		int result = assoc[piece] + 1;  // current preference cursor (index into pref_p) -> raise this until valid piece
		int max = static_cast<int>(reel_context.p()); // cast to unsigned because -1 is our error value

		while(result < max)
		{
			piece_t right = reel_context.pref(piece, result);

			if(is_right(right))
				return result - assoc[piece];

			result++;	
		}

		return -1;
	}
};

/**
 * This ordering predicate establishes a total order over EstNode::Ptrs based on their cost.
 * Lower-cost nodes pop first out of the priority_queue of the search.
 */
class greater_cost
{
public:
	bool operator() (const EstNode::Ptr& lhs, const EstNode::Ptr& rhs) const
	{
		return lhs->m_cost > rhs->m_cost;
	}
};

/**
 * Compute the heuristic distance to goal for the given assoc.
 * The result of this function is what the search is attempting to minimize.
 * From the sum of the length of all free pieces listed in lefts, excluding the not-free piece z,
 * the overlaps as specified in assoc and overmat are subtracted.
 */
static int assoc_value(const std::vector<int>& assoc, const EstContext& est_context, const ReelContext& reel_context)
{
	piece_t a = est_context.a();
	piece_t z = est_context.z();
	int length = reel_context.overlap(z, a); // revert finished-loop assumption from cost g(n)

	for(piece_t left : est_context.lefts()) {
		if(left != z)
			length += reel_context.lobs(left);

		int rank = assoc[left];
		piece_t right = reel_context.pref(left, rank);
		length -= reel_context.overlap(left, right);
	}

	return length;
}

/**
 * Create a global heuristic context object.
 * The context pointer must be passed to every call of estimate() and disposed of using destroy_context().
 * - n: Number of observed pieces in the input
 * - overmat: An n*n precomputed matrix of cost savings between pieces, stored as an array in row-major order
 * - p: Number of pieces listed in one pref list (out of the list of pref lists passed in the pref parameter)
 * - pref: An n*(n-1) precomputed list of ordered association preferences, stored as an array, one list after another
 * - lobs: An array of lengths of the observed pieces in symbols, length n
 * Return: A pointer to the created context pointer or nullptr if the operation failed
 */
extern "C" ReelContext* create_context(size_t n, int overmat[], size_t p, piece_t pref[], int lobs[]) noexcept
{
	try {
		auto context = new ReelContext(n, overmat, p, pref, lobs);
		context->log() << "Created new context, n=" << n << ", p=" << p << "\n";
		return context;
	}
	catch(const std::exception& e) {
		// Exceptions are for example std::bad_alloc.
		// Swallow error for the C interface.
		std::cerr << e.what() << "\n";
		return nullptr;
	}
}

/**
 * Destroy a global heuristic context object formerly created by create_context().
 * - context: Pointer to the object to be destroyed.
 */
extern "C" void destroy_context(const ReelContext* context) noexcept
{
	delete context;
}

/**
 * Run the heuristic search on the given partial solution.
 * - l: Length of the array of left pieces
 * - lefts: Array of left pieces
 * - a: First piece in the partial solution sequence
 * - z: Last piece in the partial solution sequence
 * - reel_context: An ReelContext object previously created with create_context
 * Return: The estimated distance to goal in symbols or -1 if an error occurred.
 */
extern "C" int estimate(size_t l, piece_t lefts[], piece_t a, piece_t z, const ReelContext* reel_context) noexcept
{
	auto log = reel_context->log();

	try {
		auto est_context = EstContext(l, lefts, a, z);
		auto leaves = std::priority_queue<EstNode::Ptr, std::vector<EstNode::Ptr>, greater_cost> ();

		log << debug_reel_context_str(*reel_context) << "\n";
		log << debug_est_context_str(est_context) << "\n";

		// initial leaves = root only
		auto root = std::make_unique<EstNode>();
		leaves.push(std::move(root));

		// resolv_steps = 100 # max iterations to try and resolve conflicts

		log << "Est search start! l=" << l << ", a=" << a << ", z=" << z << ", leaves: " << leaves.size() << "\n";
		while(!leaves.empty()) {
			// Forcibly extract the Ptr, we’re not using top() again anyway
			EstNode::Ptr node = const_cast<EstNode::Ptr&&> (leaves.top());
			leaves.pop();

			log << "Expand node " << node.get() << "...\n";

			bool search_more = node->expand(est_context, *reel_context);
			log << "Est examine <<" << debug_assoc_str(node->assoc()) << ">>  " << node->cost() << "\n";

			if(!search_more) {
				auto& assoc = node->assoc();
				return assoc_value(assoc, est_context, *reel_context);
			}

			for(EstNode::Ptr& succ : EstNode::successors(std::move(node))) {
				leaves.push(std::move(succ));
			}
		}

		// NOTE: remove this when implementing resolv step limit
		return -1; // no conflict-free solution seems possible.
	}
	catch(const std::exception& e) {
		// Exceptions are for example std::bad_alloc.
		// Swallow error for the C interface.
		log << e.what() << "\n";
		return -1;
	}
}

template<typename Container>
static std::string debug_arr2str(const Container& arr)
{
	std::ostringstream buffer;
	buffer << "{ ";

	bool first = true;
	for(const auto& elem : arr) {
		if(first)
			first = false;
		else
			buffer << ", ";

		buffer << elem;
	}

	buffer << " }";
	return buffer.str();
}

__attribute__((unused))
static std::string debug_assoc_str(const std::vector<int>& assoc)
{
	return debug_arr2str(assoc);
}

__attribute__((unused))
static std::string debug_est_context_str(const EstContext& context)
{
	std::ostringstream buffer;
	buffer << "EstContext{ ";
	buffer << "lefts=" << debug_arr2str(context.lefts()) << ", a=" << context.a() << ", z=" << context.z() << " }";
	return buffer.str();
}

__attribute__((unused))
static std::string debug_pref_str(const std::vector<std::vector<piece_t>>& pref)
{
	std::vector<std::string> pref_strings;
	auto pref2str = debug_arr2str<std::vector<piece_t>>;
	std::transform(begin(pref), end(pref), std::back_inserter(pref_strings), pref2str);
	return debug_arr2str(pref_strings);
}

__attribute__((unused))
static std::string debug_reel_context_str(const ReelContext& context)
{
	std::ostringstream buffer;
	buffer << "ReelContext{ ";
	buffer << "n=" << context.n() << ", ";
	buffer << "p=" << context.p() << ", ";
	buffer << "overmat=" << debug_arr2str(context.m_overmat) << ", ";

	// std::vector<std::string> pref_strings;
	// auto pref2str = debug_arr2str<std::vector<piece_t>>;
	// std::transform(context.m_pref.begin(), context.m_pref.end(), std::back_inserter(pref_strings), pref2str);
	buffer << "pref=" << debug_pref_str(context.m_pref) << ", ";
	buffer << "lobs=" << debug_arr2str(context.m_lobs) << " }";
	return buffer.str();
}

/*******************************************************************************
 * Self-contained unit tests - compile this module as a standalone executable. *
 *******************************************************************************/

void test1();
void test2();

/**
 * This main function turns this module into a stand-alone unit test.
 */
int main()
{
	test1();
	test2();
}

/**
 * Test 1
 * A very basic test with 4 obs pieces & ambiguous pref
 */
void test1()
{
	// problem environment
	size_t n = 4;
	__attribute__((unused)) const char* obs[] = {"abcd", "decd", "cdfa", "bcd"}; // for reference
	int overmat[] = { 0, 1, 2, 0,  0, 0, 2, 0,  1, 0, 0, 0,  0, 1, 2, 0 };
	size_t p = 2;
	piece_t pref[] = { 2, 1,  2, 0,  0, 1,  -1, -1 };
	int lobs[] = { 4, 4, 4, 3 };

	std::cerr << "== Test 1 ==\n";
	std::cerr << "obs=" << debug_arr2str(std::vector<const char*>(obs, obs+n)) << "\n";

	const ReelContext* context = create_context(n, overmat, p, pref, lobs);

	// heuristic environment
	size_t l = 3;
	piece_t lefts[] = { 0, 1, 2 };
	piece_t a = 0;
	piece_t z = 0;

	int expected = 4;
	int actual = estimate(l, lefts, a, z, context);

	if(expected == actual)
		std::cerr << "Result OK\n";
	else
		std::cerr << "Result FAILED: expected=" << expected << ", actual=" << actual << "\n";

	destroy_context(context);
}

/**
 * Test 2
 * A test derived from the r30_2 reel test case.
 */
void test2()
{
	// problem environment
	size_t n = 19;
	__attribute__((unused)) const char* obs[] = {"03b95b7", "076190", "0b72", "3b920031", "48989a78", "56a2038", "689", "6a0b7", "75a95", "7703b9", "782075a", "83bb256", "89a78207", "8ba56748", "9163", "9200317", "a06a0b", "a20385", "b7207"};
	int overmat[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2,
		1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 5, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 5, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0,
		0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2,
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
		0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
	};

	size_t p = 18;
	piece_t pref[] = {
		18, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
		0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		18, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		15, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
		12, 10, 11, 13, 0, 1, 2, 3, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18,
		17, 11, 12, 13, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 14, 15, 16, 18,
		12, 14, 15, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 16, 17, 18,
		2, 18, 8, 9, 10, 0, 1, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17,
		5, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		0, 3, 14, 15, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18,
		8, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 18,
		5, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18,
		10, 1, 8, 9, 0, 2, 3, 4, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18,
		4, 11, 12, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18,
		3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18,
		8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 16, 17, 18,
		7, 2, 18, 0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17,
		5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
		1, 8, 9, 10, 0, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17
	};
	int lobs[] = { 7, 6, 4, 8, 8, 7, 3, 5, 5, 6, 7, 7, 8, 8, 4, 7, 6, 6, 5 };

	std::cerr << "== Test 2 ==\n";
	std::cerr << "obs=" << debug_arr2str(std::vector<const char*>(obs, obs+n)) << "\n";

	const ReelContext* context = create_context(n, overmat, p, pref, lobs);

	// heuristic environment
	size_t l = 17;
	piece_t lefts[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18 };
	piece_t a = 0;
	piece_t z = 4;

	// partial solution = 0, 13, 4. Length = 7 + 8 + 8 - 2 = 21, est <= (70-21) = 49
	int expected = 49;
	int actual = estimate(l, lefts, a, z, context);

	if(expected >= actual)
		std::cerr << "Result OK (" << actual << "<=" << expected << ")\n";
	else
		std::cerr << "Result FAILED: expected=" << expected << " < actual=" << actual << "\n";
}
